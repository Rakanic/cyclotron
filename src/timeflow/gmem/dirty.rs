//! Dirty L0 lines and their writebacks into the cluster L1 (policy `l0_writeback_model`).
//!
//! Radiance's L0d (a Rocket NonBlockingDCache) is write-back and write-allocate. When a fill evicts a line the core
//! has written, the cache releases it; TLCToTLULNode turns the release into PutFullData to the cluster L1, one per L1
//! line (a 64 B L0d line is two 32 B Puts). The Puts go through the same L1 input adapter as the core's misses, so
//! they take its in-flight slots (`l1_inflight_limit`). Nearly all of them hit in the L1 (fa_mxfp8_sr FSDB: 3,204 of
//! 3,240), so a Put holds its slot for a fixed `l0_writeback_cycles` (the measured mean) instead of travelling the
//! graph, and no warp waits for it. An L0 flush writes back every dirty line the same way.

use std::collections::{HashSet, VecDeque};

use crate::timeq::Cycle;

pub(crate) struct L0Writebacks {
    dirty: Vec<HashSet<u64>>,         // [core] dirty L0 lines
    pending: Vec<usize>,              // [cluster] Puts waiting for an L1 slot
    holding: Vec<VecDeque<Cycle>>,    // [cluster] when the slots Puts hold free (in order: the hold is fixed)
    puts_per_line: usize,
    hold: Cycle,
}

impl L0Writebacks {
    pub(crate) fn new(cores: usize, clusters: usize, puts_per_line: usize, hold: Cycle) -> Self {
        Self {
            dirty: vec![HashSet::new(); cores],
            pending: vec![0; clusters],
            holding: vec![VecDeque::new(); clusters],
            puts_per_line: puts_per_line.max(1),
            hold: hold.max(1),
        }
    }

    /// A store of `core` wrote L0 line `line`
    pub(crate) fn mark_dirty(&mut self, core: usize, line: u64) {
        if let Some(d) = self.dirty.get_mut(core) {
            d.insert(line);
        }
    }

    /// `core`'s L0 dropped `line` (a fill evicted it): write it back if it is dirty
    pub(crate) fn evicted(&mut self, core: usize, cluster: usize, line: u64) {
        if self.dirty.get_mut(core).map_or(false, |d| d.remove(&line)) {
            if let Some(p) = self.pending.get_mut(cluster) {
                *p += self.puts_per_line;
            }
        }
    }

    /// `core`'s L0 was flushed: every dirty line is written back
    pub(crate) fn flushed(&mut self, core: usize, cluster: usize) {
        let n = self.dirty.get_mut(core).map_or(0, |d| std::mem::take(d).len());
        if let Some(p) = self.pending.get_mut(cluster) {
            *p += n * self.puts_per_line;
        }
    }

    /// The L1 slots `cluster`'s Puts free at `now`
    pub(crate) fn expire(&mut self, cluster: usize, now: Cycle) -> usize {
        let q = &mut self.holding[cluster];
        let mut n = 0;
        while q.front().map_or(false, |&t| t <= now) {
            q.pop_front();
            n += 1;
        }
        n
    }

    pub(crate) fn pending(&self, cluster: usize) -> usize {
        self.pending[cluster]
    }

    /// One of `cluster`'s pending Puts takes an L1 slot at `now`
    pub(crate) fn start(&mut self, cluster: usize, now: Cycle) {
        self.pending[cluster] -= 1;
        self.holding[cluster].push_back(now + self.hold);
    }
}
