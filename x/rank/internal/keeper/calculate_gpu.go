// +build cuda

package keeper

import (
	"github.com/cybercongress/cyberd/x/link"
	"github.com/cybercongress/cyberd/x/rank/internal/types"
	"github.com/tendermint/tendermint/libs/log"
	"reflect"
	"math"
	"sync"
	"time"
)

/*
#cgo CFLAGS: -I/usr/lib/
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcbdrank -lcudart
#include "cbdrank.h"
*/
import "C"

func calculateRankGPU(ctx *types.CalculationContext, logger log.Logger) []float64 {
	start := time.Now()
	if ctx.GetCidsCount() == 0 {
		return make([]float64, 0)
	}

	tolerance := ctx.GetTolerance()
	dampingFactor := ctx.GetDampingFactor()

	outLinks := ctx.GetOutLinks()

	cidsCount := ctx.GetCidsCount()
	stakesCount := len(ctx.GetStakes())

	linksCount := uint64(0)

	linksCount2 := uint64(0)

	rank := make([]float64, cidsCount)
	inLinksCount := make([]uint32, cidsCount)
	outLinksCount := make([]uint32, cidsCount)

	inLinksCount2 := make([]uint32, cidsCount)
	outLinksCount2 := make([]uint32, cidsCount)

	inLinksOuts := make([]uint64, 0)
	inLinksUsers := make([]uint64, 0)
	outLinksUsers := make([]uint64, 0)

	inLinksOuts2 := make([]uint64, 0)
	inLinksUsers2 := make([]uint64, 0)
	outLinksUsers2 := make([]uint64, 0)

	// todo reduce size of stake by passing only participating in linking stakes.
	// todo need to investigate why GetStakes returns accounts with one missed index, think this is goes from some module
	stakes := make([]uint64, stakesCount+10)
	for acc, stake := range ctx.GetStakes() {
		stakes[uint64(acc)] = stake
	}

	// ___________________________________________________

	ch := make(chan int64, 100000)
	var wg sync.WaitGroup
	var lock1 sync.Mutex
	var lock2 sync.Mutex
	wg.Add(int(cidsCount))

	// the worker's function
	f := func(i int64) {
		defer wg.Done()
		if inLinks, sortedCids, ok := ctx.GetSortedInLinks(link.CidNumber(i)); ok {
			for _, cid := range sortedCids {
				inLinksCount[i] += uint32(len(inLinks[cid]))
				for acc := range inLinks[cid] {
					lock2.Lock()
					inLinksOuts = append(inLinksOuts, uint64(cid))
					inLinksUsers = append(inLinksUsers, uint64(acc))
					lock2.Unlock()
				}
			}
			linksCount += uint64(inLinksCount[i])
		}

		if outLinks, ok := outLinks[link.CidNumber(i)]; ok {
			for _, accs := range outLinks {
				outLinksCount[i] += uint32(len(accs))
				for acc := range accs {
					lock1.Lock()
					outLinksUsers = append(outLinksUsers, uint64(acc))
					lock1.Unlock()
				}
			}
		}
	}

	countWorkers := int64(math.Min(10000, float64(cidsCount)))

	// here the workers start
	for i:=int64(0); i < countWorkers; i++ {
		go func() {
			var cid int64
			for {
				cid = <- ch
				f(cid)
			}
		}()
	}

	// data is added to the channel for workers
	for i := int64(0); i < cidsCount; i++ {
		ch <- i
	}

	// waiting for a while all workers will finish work
	wg.Wait()

	// ___________________________________________________

	for i := int64(0); i < cidsCount; i++ {

		if inLinks, sortedCids, ok := ctx.GetSortedInLinks(link.CidNumber(i)); ok {
			for _, cid := range sortedCids {
				inLinksCount2[i] += uint32(len(inLinks[cid]))
				for acc := range inLinks[cid] {
					inLinksOuts2 = append(inLinksOuts2, uint64(cid))
					inLinksUsers2 = append(inLinksUsers2, uint64(acc))
				}
			}
			linksCount2 += uint64(inLinksCount2[i])
		}

		if outLinks, ok := outLinks[link.CidNumber(i)]; ok {
			for _, accs := range outLinks {
				outLinksCount2[i] += uint32(len(accs))
				for acc := range accs {
					outLinksUsers2 = append(outLinksUsers2, uint64(acc))
				}
			}
		}
	}

	// ___________________________________________________

	if (!reflect.DeepEqual(linksCount, linksCount2)) {
		logger.Error("linkCount not equal")
	}

	if (!reflect.DeepEqual(inLinksCount,inLinksCount2)) {
		logger.Error("inLinksCount not equal")
	}

	if (!reflect.DeepEqual(outLinksCount, outLinksCount2)) {
		logger.Error("outLinksCount not equal")
	}

	if (!reflect.DeepEqual(inLinksOuts, inLinksOuts2)) {
		logger.Error("inLinksOuts2 not equal")
	}

	if (!reflect.DeepEqual(inLinksUsers, inLinksUsers2)) {
		logger.Error("inLinksUsers not equal")
	}

	if (!reflect.DeepEqual(outLinksUsers, outLinksUsers2)) {
		logger.Error("outLinksUsers not equal")
	}

	// ___________________________________________________

	/* Convert to C types */
	cStakes := (*C.ulong)(&stakes[0])

	cStakesSize := C.ulong(len(stakes))
	cCidsSize := C.ulong(len(inLinksCount))
	cLinksSize := C.ulong(len(inLinksOuts))

	cInLinksCount := (*C.uint)(&inLinksCount[0])
	cOutLinksCount := (*C.uint)(&outLinksCount[0])

	cInLinksOuts := (*C.ulong)(&inLinksOuts[0])
	cInLinksUsers := (*C.ulong)(&inLinksUsers[0])
	cOutLinksUsers := (*C.ulong)(&outLinksUsers[0])

	cDampingFactor := C.double(dampingFactor)
	cTolerance := C.double(tolerance)

	logger.Info("Rank: data for gpu preparation", "time", time.Since(start))

	start = time.Now()
	cRank := (*C.double)(&rank[0])
	C.calculate_rank(
		cStakes, cStakesSize, cCidsSize, cLinksSize,
		cInLinksCount, cOutLinksCount,
		cInLinksOuts, cInLinksUsers, cOutLinksUsers,
		cRank, cDampingFactor, cTolerance,
	)
	logger.Info("Rank: gpu calculations", "time", time.Since(start))

	return rank
}

