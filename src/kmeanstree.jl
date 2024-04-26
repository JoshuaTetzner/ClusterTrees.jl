module KMeansTrees

using ParallelKMeans
using ClusterTrees
using StaticArrays
using LinearAlgebra

struct Data{F,T}
    center::SVector{3,F}
    radius::F
    values::Vector{T}
end

struct KMeansSettings
    max_iters::Int
    n_threads::Int
end

function KMeansSettings(; max_iters=100, n_threads=1)
    return KMeansSettings(max_iters, n_threads)
end

struct KMNode{D}
    node::ClusterTrees.PointerBasedTrees.Node{D}
    height::Int
end

struct KMeansTree{D} <: ClusterTrees.PointerBasedTrees.APBTree
    nodes::Vector{KMNode{D}}
    root::Int
    num_elements::Int
    levels::Vector{Int}
end

function KMeansTree(num_elements; center=SVector(0.0, 0.0, 0.0), radius=0.0, data=Vector(1:num_elements))
    root = KMNode(
        ClusterTrees.PointerBasedTrees.Node(Data(center, radius, data), 0, 0, 0, 2), 0
    )

    return KMeansTree([root], 1, num_elements, Int[1])
end

ClusterTrees.root(tree::KMeansTree{D}) where {D} = tree.root
ClusterTrees.data(tree::KMeansTree{D}, node) where {D} = tree.nodes[node].node.data
ClusterTrees.parent(tree::KMeansTree{D}, node) where {D} = tree.nodes[node].node.parent
ClusterTrees.PointerBasedTrees.nextsibling(tree::KMeansTree, node) =
    tree.nodes[node].node.next_sibling
ClusterTrees.PointerBasedTrees.firstchild(tree::KMeansTree, node) =
    tree.nodes[node].node.first_child

function value(tree, node::Int)
    if !ClusterTrees.haschildren(tree, node)
        return tree.nodes[node].node.data.values
    else
        values = Int[]
        for leave in ClusterTrees.leaves(tree, node)
            append!(values, tree.nodes[leave].node.data.values)
        end
        return values
    end
end


function child!(tree::KMeansTree{D}, state, destination) where {D}
    maxlevel, num_children, points, kmeans_sttings, nmin = destination
    parent_node_idx, level, sibling_idx, point_idcs = state

    kmcluster = ParallelKMeans.kmeans(
        points[:, point_idcs],
        num_children;
        max_iters=kmeans_sttings.max_iters,
        n_threads=kmeans_sttings.n_threads,
    )

    sorted_point_idcs = zeros(Int, length(point_idcs) + 1, num_children)

    for (index, value) in enumerate(kmcluster.assignments)
        sorted_point_idcs[1, value] += 1
        sorted_point_idcs[sorted_point_idcs[1, value] + 1, value] = point_idcs[index]
    end

   
    isnmin = sorted_point_idcs[1, 1] < nmin 
    for sidx in 2:num_children
        isnmin = isnmin || sorted_point_idcs[1, sidx] < nmin
    end
    if isnmin || level > maxlevel
        append!(tree.nodes[parent_node_idx].node.data.values, point_idcs)
        return 0
    end

    center = SVector{3,Float64}([kmcluster.centers[j, 1] for j in 1:3])
    radius = maximum(
        norm.(
            eachcol(
                points[:, sorted_point_idcs[2:(sorted_point_idcs[1, 1] + 1), 1]] .-
                kmcluster.centers[:, 1],
            )
        ),
    )
   
    push!(
        tree.nodes,
        KMNode(
            ClusterTrees.PointerBasedTrees.Node(
                Data(center, radius, Int[]), num_children, 0, parent_node_idx, 0
            ),
            level,
        ),
    )

    node_idx = length(tree.nodes)
    level >= length(tree.levels) && resize!(tree.levels, level+1)
    tree.levels[level+1] = node_idx
    state_sibling = (
        parent_node_idx,
        level,
        sibling_idx + 1,
        point_idcs,
        sorted_point_idcs,
        kmcluster,
    )
    tree.nodes[node_idx].node.next_sibling = sibling!(tree, state_sibling, destination)

    # Check if more than one node is left
    state_child = (
        node_idx, level + 1, 1, sorted_point_idcs[(2:(sorted_point_idcs[1, 1] + 1)), 1]
    )
    tree.nodes[node_idx].node.first_child = child!(tree, state_child, destination)

    return node_idx

end

function sibling!(tree::KMeansTree{D}, state, destination) where {D}
    maxlevel, num_children, points, kmeans_sttings, nmin = destination
    parent_node_idx, level, sibling_idx, point_idcs, sorted_point_idcs, kmcluster = state

    # Enough siblings?
    sibling_idx > num_children && return 0

    center = SVector{3,Float64}([kmcluster.centers[j, sibling_idx] for j in 1:3])
    radius = maximum(
        norm.(
            eachcol(
                points[
                    :,
                    sorted_point_idcs[
                        2:(sorted_point_idcs[1, sibling_idx] + 1), sibling_idx
                    ],
                ] .- kmcluster.centers[:, sibling_idx],
            )
        ),
    )

    push!(
        tree.nodes,
        KMNode(
            ClusterTrees.PointerBasedTrees.Node(
                Data(center, radius, Int[]), num_children, 0, parent_node_idx, 0
            ),
            level,
        ),
    )

    node_idx = length(tree.nodes)
    level >= length(tree.levels) && resize!(tree.levels, level+1)
    tree.levels[level+1] = node_idx

    state_sibling = (
        parent_node_idx,
        level,
        sibling_idx + 1,
        point_idcs,
        sorted_point_idcs,
        kmcluster,
    )
    tree.nodes[node_idx].node.next_sibling = sibling!(tree, state_sibling, destination)

    # Check if more than one node is left
    state_child = (
        node_idx,
        level + 1,
        1,
        sorted_point_idcs[(2:(sorted_point_idcs[1, sibling_idx] + 1)), sibling_idx],
    )
    tree.nodes[node_idx].node.first_child = child!(tree, state_child, destination)

    return node_idx
end

end