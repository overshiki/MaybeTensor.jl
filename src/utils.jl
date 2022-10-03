######## utils for NSVector ########
len(vs::Vector{T}) where T = length(vs)
select(v::Vector{T}, index::Int) where T = Vector{T}([v[index]])
select(v::Nothing, index::Int) = nothing
select_element(v::Vector{T}, index::Int) where T = bind(select(v, index), xs->xs[1])

unsqueeze(a::Array) = begin 
    nsize = foldl(append!, size(a); init=[1])
    return reshape(a, Tuple(nsize))
end
fullstack(vs::Vector{T}) where T<:Vector = begin
    return reduce(vcat, map(unsqueeze, map(fullstack, vs)))
end
fullstack(vs::Vector{T}) where T<:Real = vs

concat(alist::Vector{Vector{T}}) where T = foldl(++, alist; init=T[])

cleanArray(vs::Array) = begin 
    @assert size(vs, ndims(vs))==1
    return dropdims(vs;dims=ndims(vs))
end

cleanNSVector(vs::Vector{Vector{T}}) where T<:Vector = map(cleanNSVector, vs)
cleanNSVector(vs::Vector{Vector{T}}) where T<:Real = concat(vs)

Empty(ns::Vector{T}) where T<:Real = T[]
(Empty(ns::Vector{Vector{T}})::Vector{Vector{T}}) where T<:Real = map(Empty, ns)

######## Monad for Union{Vector, Nothing}########
"""Monad chain: ((Ma, Ma), (Ma, Ma)-> Ma) -> Ma"""
chain(x::Union{Vector{T1}, Nothing}, y::Union{Vector{T2}, Nothing}, f::Function) where T1 where T2 = begin 
    x isa Nothing && return y 
    y isa Nothing && return x 
    return f(x, y)
end
"""Monad bind: Ma, (a->Mb) -> Mb"""
bind(x::Union{Vector{T}, Nothing}, f::Function) where T = begin
    x isa Nothing && return x 
    return f(x)
end
broadcast_bind(x::Union{Vector{T}, Nothing}, f::Function) where T = bind(x, x->map(f, x))

"""f: Ma, Int -> Mb"""
indexhook_bind(x::Union{Vector{T}, Nothing}, f::Function) where T = begin 
    func(x) = map(1:len(x)) do i 
        f(x[i], i)
    end 
    return bind(x, func)
end

broadcast_chain(x::Union{Vector{T1}, Nothing}, y::Union{Vector{T2}, Nothing}, f::Function) where T1 where T2 = begin
    x isa Nothing && return y 
    y isa Nothing && return x 
    @assert len(x)==len(y)
    return chain(x, y, (x,y)->map(t->f(t...), zip(x, y)))
end


######## utils for Vector ########

(++)(xlist::Vector{T}, ylist::Vector{T}) where T = begin 
    clist = T[]
    append!(clist, xlist)
    append!(clist, ylist)
    return clist
end 

binary2maybe(binary_pred::Function) = begin 
    func(x) = begin 
        binary_pred(x) && return [x] 
        return nothing 
    end
    return func
end

"""pred: a->Ma"""
filterl(pred::Function, vs::Vector{T}) where T = begin
    maybe_pred = binary2maybe(pred) 
    return foldl((x,y)->chain(x, maybe_pred(y), ++), vs; init=T[])
end

filterl(pred::Vector{Bool}, vs::Vector{T}) where T = begin
    binaryZip2maybe(y::Tuple{Bool, T}) = begin 
        b, v = y 
        b && return [v] 
        return nothing
    end
    @assert len(pred)==len(vs)
    nvs = collect(zip(pred, vs))
    return foldl((x,y)->chain(x, binaryZip2maybe(y), ++), nvs; init=T[])
end