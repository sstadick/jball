# JBall

Powers https://www.patreon.com/WaiverWireScience

**NOTE** work in progress as the original is ported from different data sources and from Python. 

## Synopsis

This takes stats from last years player, clusters last years players, then projects this years players onto last years players clusters to see who looks comparable.

**Strictly Speaking these are more groupings than rankings**

### More Details

Stats from last years players are averaged over all the games they played, then run through PCA to reduce dimensionality. Last years PCA transformed results are then clustered with Kmeans.

This years players stats, averaged over the last 4 (or max) games are then projected into the same PCA space, and then added to last years clusters.

Rankings are done by taking the average fantasy points per game of last years players within a cluster to rank clusters, then players within each cluster are ranked based on average fantasy points per game.

Fantasy points shown are .5 PPR.

### ELI5

Look in the top 2ish clusters, find players there on waivers and if they have a compelling story, they are probably worth a pickup.

You can look up last years players that they are similar to by finding the cluster_id of a player in the tab for last year to see how they compare.

## Run

```
julia
]
activate .
using Revise
using JBall
JBall.run(:wr)
JBall.run(:te)
JBall.run(:rb)
JBall.run(:qb)
```