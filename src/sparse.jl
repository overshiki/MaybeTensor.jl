


# copyline(alist::Vector{Int}, 
#         empty_list::Vector{Int}, 
#         index::Int)::Vector{Int} = empty_list ++ [alist[index]]

# copyline(alist::Vector{Vector{Int}}, 
#         empty_list::Vector{Vector{Int}}, 
#         index::Int)::Vector{Vector{Int}} = map(z->copyline(z..., index), zip(alist, empty_list))

# prune_indices(indices::Vector{Vector{Int}}, index::Int) = begin 
#     x, xs = indices[1], indices[2:end]
#     let nxs = Empty(xs)
#         for i in 1:len(x)
#             if x[i]==index 
#                 nxs = copyline(xs, nxs, i)
#             end
#         end
#         return nxs
#     end
# end



prune_indices(indices::Vector{Vector{Int}}, index::Int) = begin 
    len(indices)==0 && return indices
    x, xs = indices[1], indices[2:end]
    nind = fromNSVector(xs) |> transpose
    nind = filterl(map(xi->xi==index, x), values(nind))
    len(nind)==0 && return Vector{Int}[]
    return nind |> NSTensor |> transpose |> toNSVector |> cleanNSVector
end



"""build new NSTensor from the Empty tensor and the coo information, in a recursive way"""
(copyfrom(indices::Vector{Vector{Int}}, values::Vector{T}, ns::NSTensor{T2})::NSTensor) where T<:AbstractMaybeTensor where T2<:NSTensor = begin 
    indexhook_bind(ns, (nns, i)->copyfrom(prune_indices(indices, i), values, nns))
end

copy_with_default(indices::Vector{Int}, values::Vector{T}, index::Int) where T<:MaybeLeafTensor = begin 
    index in indices && return values[index]
    return Empty(T)
end
(copyfrom(indices::Vector{Vector{Int}}, values::Vector{T}, ns::NSTensor{T2})::NSTensor) where T<:AbstractMaybeTensor where T2<:MaybeLeafTensor = begin 
    @assert len(indices)<=1
    if len(indices)==0 
        return ns
    else
        indices = indices[1]
        return indexhook_bind(ns, (nns, i)->copy_with_default(indices, values, i))
    end
end

(coo_tensor(indices::Vector{Vector{Int}}, values::Vector{T}, shape::Vector{Int})::NSTensor) where T<:AbstractMaybeTensor = begin 
    ns = Empty(shape, T)
    return copyfrom(indices, values, ns)
end