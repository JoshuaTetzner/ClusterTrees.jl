using ClusterTrees
using CompScienceMeshes
using LinearAlgebra

function updatestate(block_tree::ClusterTrees.BlockTrees.BlockTree{T}, chd) where {T}
    d = ClusterTrees.data(block_tree, chd)
    test_center = d[1].center
    trial_center = d[2].center
    test_radius = d[1].radius
    trial_radius = d[2].radius
    return ((test_center, test_radius), (trial_center, trial_radius))
end

function listnearfarinteractions(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    block,
    state,
    nears::Vector{Tuple{Int,Int}},
    fars::Vector{Vector{Tuple{Int,Int}}},
    level::Int,
) where {T}
    isfar(state) && (push!(fars[level], block); return nothing)
    !ClusterTrees.haschildren(block_tree, block) && (push!(nears, block); return nothing)
    for chd in ClusterTrees.children(block_tree, block)
        chd_state = updatestate(block_tree, chd)
        listnearfarinteractions(block_tree, chd, chd_state, nears, fars, level + 1)
    end
end

function isfar(state)
    η = 2
    test_state, trial_state = state
    test_center, test_radius = test_state
    trial_center, trial_radius = trial_state

    center_dist = norm(test_center - trial_center)
    dist = center_dist - (test_radius + trial_radius)
    
    if 2 * max(test_radius, trial_radius) <= η * (dist) && dist != 0
        return true
    else
        return false
    end
end

function computeinteractions(tree::ClusterTrees.BlockTrees.BlockTree{T}) where {T}
    nears = Tuple{Int,Int}[]
    num_levels = length(tree.test_cluster.levels)
    fars = [Tuple{Int,Int}[] for l in 1:num_levels]

    root_state = (
        (
            tree.test_cluster.nodes[1].node.data.center,
            tree.test_cluster.nodes[1].node.data.radius,
        ),
        (
            tree.trial_cluster.nodes[1].node.data.center,
            tree.trial_cluster.nodes[1].node.data.radius,
        ),
    )
    root_level = 1

    listnearfarinteractions(
        tree, ClusterTrees.root(tree), root_state, nears, fars, root_level
    )

    return nears, fars
end

##

points = meshsphere(1.0, 0.2).vertices[2:end]
tree = ClusterTrees.KMeansTrees.KMeansTree(length(points))
destination = (
    5,
    2, 
    reshape([point[i] for point in points for i in 1:3], (3, length(points))),
    ClusterTrees.KMeansTrees.KMeansSettings(),
    2
)
state = (1, 1, 1, Vector(1:length(points)))
ClusterTrees.KMeansTrees.child!(tree, state, destination)

block_tree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
nears, fars = computeinteractions(block_tree)