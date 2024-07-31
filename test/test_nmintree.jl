using ClusterTrees
using StaticArrays
using Test


# KMeansTree

points = [
    0.0 0.0 2.0 2.0;
    0.5 0.0 0.5 0.0;
    0.0 0.0 0.0 0.0
]

tree = ClusterTrees.NminTrees.NminTree(size(points, 2))
treeoptions = ClusterTrees.NminTrees.KMeansTreeOptions()
destination = (1, 5)
state = (1, SVector(0.0, 0.0, 0.0), 1.0, 1, 1, Vector(1:size(points, 2)), points)
ClusterTrees.NminTrees.child!(tree, treeoptions, state, destination)

@test length(tree.levels) == 3
@test length(ClusterTrees.NminTrees.value(tree, 1)) == 4
@test length(ClusterTrees.NminTrees.value(tree, 4)) == 1
@test ClusterTrees.NminTrees.root(tree) == 1
@test ClusterTrees.NminTrees.data(tree, 1).values == []
@test ClusterTrees.parent(tree, 2) == 1

##

points = rand(Float64, 3, 100)

nmin = 5
tree = ClusterTrees.NminTrees.NminTree(size(points, 2))
treeoptions = ClusterTrees.NminTrees.KMeansTreeOptions()
destination = (nmin, 10)
state = (1, SVector(0.0, 0.0, 0.0), 1.0, 1, 1, Vector(1:size(points, 2)), points)
ClusterTrees.NminTrees.child!(tree, treeoptions, state, destination)

for leaf in ClusterTrees.leaves(tree, 1)
    @test length(ClusterTrees.NminTrees.value(tree, leaf)) >= 5
end

##

maxlevel = 3
tree = ClusterTrees.NminTrees.NminTree(size(points, 2))
treeoptions = ClusterTrees.NminTrees.KMeansTreeOptions()
destination = (1, 3)
state = (1, SVector(0.0, 0.0, 0.0), 1.0, 1, 1, Vector(1:size(points, 2)), points)
ClusterTrees.NminTrees.child!(tree, treeoptions, state, destination)

# rootlevel is zeroth level
@test length(tree.levels) == maxlevel+1

## BoxTree
points = [
    SVector(0.5, 0.5, 0.5)
]


hs, ct = ClusterTrees.NminTrees.boundingbox(points)
tree = ClusterTrees.NminTrees.NminTree(size(points, 2))
treeoptions = ClusterTrees.NminTrees.BoxTreeOptions()
destination = (2, 5)
state = (1, ct, sqrt(3)*hs, 1, 1, Vector(1:length(points)), points)
ClusterTrees.NminTrees.child!(tree, treeoptions, state, destination)

@test tree.nodes[1].node.first_child == 0

for leaf in ClusterTrees.leaves(tree, 1)
    @test leaf == 1
end

##

points = [
    SVector(0.5, 0.5, 0.5),
    SVector(0.5, 0.5, -0.5),
    SVector(0.5, -0.5, 0.5),
    SVector(0.5, -0.5, -0.5),
    SVector(-0.5, -0.5, 0.5),
    SVector(-0.5, -0.5, -0.5),
    SVector(-0.5, 0.5, 0.5),
    SVector(-0.5, 0.5, -0.5),
]


hs, ct = ClusterTrees.NminTrees.boundingbox(points)
tree = ClusterTrees.NminTrees.NminTree(size(points, 2))
treeoptions = ClusterTrees.NminTrees.BoxTreeOptions()
destination = (1, 5)
state = (1, ct, sqrt(3)*hs, 1, 1, Vector(1:length(points)), points)
ClusterTrees.NminTrees.child!(tree, treeoptions, state, destination)
for leaf in ClusterTrees.leaves(tree, 1)
    @test length(ClusterTrees.NminTrees.value(tree, leaf)) == 1
end
@test length(tree.levels) == 2
@test ClusterTrees.NminTrees.root(tree) == 1
@test ClusterTrees.parent(tree, 2) == 1

##

points = [
    SVector(0.5, 0.5, 0.5),
    SVector(0.2, 0.2, 0.2),
    SVector(0.5, 0.5, -0.5),
    SVector(0.5, -0.5, 0.5),
    SVector(0.5, -0.5, -0.5),
    SVector(-0.5, -0.5, 0.5),
    SVector(-0.5, -0.5, -0.5),
    SVector(-0.5, 0.5, 0.5),
    SVector(-0.5, 0.5, -0.5),
]

hs, ct = ClusterTrees.NminTrees.boundingbox(points)
tree = ClusterTrees.NminTrees.NminTree(size(points, 2))
treeoptions = ClusterTrees.NminTrees.BoxTreeOptions()
destination = (1, 5)
state = (1, ct, sqrt(3)*hs, 1, 1, Vector(1:length(points)), points)
ClusterTrees.NminTrees.child!(tree, treeoptions, state, destination)
tree.levels
@test ClusterTrees.NminTrees.value(tree, 9) == [2, 1]
@test ClusterTrees.NminTrees.value(tree, 2) == ClusterTrees.NminTrees.data(tree, 2).values
@test length(ClusterTrees.NminTrees.value(tree, 9)) == 2
@test length(tree.levels) == 3

##

N = 100
points = [@SVector rand(3) for i = 1:N]

hs, ct = ClusterTrees.NminTrees.boundingbox(points)
tree = ClusterTrees.NminTrees.NminTree(size(points, 2))
treeoptions = ClusterTrees.NminTrees.BoxTreeOptions()
destination = (5, 10)
state = (1, ct, sqrt(3)*hs, 1, 1, Vector(1:length(points)), points)
ClusterTrees.NminTrees.child!(tree, treeoptions, state, destination)

for leaf in ClusterTrees.leaves(tree, 1)
    @test length(ClusterTrees.NminTrees.value(tree, leaf)) >= 5
end


##
points = [
    SVector(0.51, 0.51, 0.51),
    SVector(0.27, 0.27, 0.27),
    SVector(0.25, 0.25, 0.25),
    SVector(0.51, 0.51, -0.51),
    SVector(0.51, -0.51, 0.51),
    SVector(0.51, -0.51, -0.51),
    SVector(-0.51, -0.51, 0.51),
    SVector(-0.51, -0.51, -0.51),
    SVector(-0.51, 0.51, 0.51),
    SVector(-0.51, 0.51, -0.51),
    SVector(0.25, 0.25, -0.25),
    SVector(0.25, -0.25, -0.25),
    SVector(0.25, -0.25, 0.25),
    SVector(-0.25, -0.25, -0.25),
    SVector(-0.25, -0.25, 0.25),
    SVector(-0.25, 0.25, -0.25),
    SVector(-0.25, 0.25, 0.25),

]

hs, ct = ClusterTrees.NminTrees.boundingbox(points)
tree = ClusterTrees.NminTrees.NminTree(size(points, 2))
treeoptions = ClusterTrees.NminTrees.BoxTreeOptions()
destination = (1, 5)
state = (1, ct, sqrt(3)*hs, 1, 1, Vector(1:length(points)), points)
ClusterTrees.NminTrees.child!(tree, treeoptions, state, destination)

for leaf in ClusterTrees.leaves(tree, 1)
    length(ClusterTrees.NminTrees.value(tree, leaf)) == 1
end

@test length(ClusterTrees.NminTrees.value(tree, 1)) == length(points)