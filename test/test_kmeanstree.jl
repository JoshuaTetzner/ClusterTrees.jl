using ClusterTrees
using StaticArrays
using Test


points = [
    0.0 0.0 2.0 2.0;
    0.5 0.0 0.5 0.0;
    0.0 0.0 0.0 0.0
]

tree = ClusterTrees.KMeansTrees.KMeansTree(size(points, 2))
destination = (4, 2, points,  ClusterTrees.KMeansTrees.KMeansSettings(), 1)
state = (1, 1, 1, Vector(1:size(points, 2)))
ClusterTrees.KMeansTrees.child!(tree, state, destination)

@test length(tree.levels) == 3
@test length(ClusterTrees.KMeansTrees.value(tree, 1)) == 4
@test length(ClusterTrees.KMeansTrees.value(tree, 4)) == 1
@test ClusterTrees.KMeansTrees.root(tree) == 1
@test ClusterTrees.KMeansTrees.data(tree, 1).values == [1, 2, 3, 4]
@test ClusterTrees.parent(tree, 2) == 1

##

points = rand(Float64, 3, 100)

nmin = 5
tree = ClusterTrees.KMeansTrees.KMeansTree(size(points, 2))
destination = (10, 2, points,  ClusterTrees.KMeansTrees.KMeansSettings(), nmin)
state = (1, 1, 1, Vector(1:size(points, 2)))
ClusterTrees.KMeansTrees.child!(tree, state, destination)

for leaf in ClusterTrees.leaves(tree, 1)
    @test length(ClusterTrees.KMeansTrees.value(tree, leaf)) >= 5
end

##

maxlevel = 3
tree = ClusterTrees.KMeansTrees.KMeansTree(size(points, 2))
destination = (maxlevel, 2, points,  ClusterTrees.KMeansTrees.KMeansSettings(), 1)
state = (1, 1, 1, Vector(1:size(points, 2)))
ClusterTrees.KMeansTrees.child!(tree, state, destination)

# rootlevel is zeroth level
@test length(tree.levels) == maxlevel+1
