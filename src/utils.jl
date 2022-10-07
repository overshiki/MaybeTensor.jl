######## utils for NSVector ########
(++)(xlist::Vector{T}, ylist::Vector{T}) where T = begin 
    clist = T[]
    append!(clist, xlist)
    append!(clist, ylist)
    return clist
end 


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

########Treat Union{Vector, Nothing} as Monad Maybe########
"""
Note that the Monad defined here is slightly different from Haskell's definition of 
    data Maybe a = Just a | Nothing 
Instead, it is a union type Union{Vector, Nothing}. The difference is that, when a function returns a Vector, we do not need to wrap it with Just notation anymore. I think this is actually more convinient to use, but may not be safe enough though.
"""

const Maybe{T} = Union{T, Nothing}


"""Monad bind: M [a] -> ([a] -> b) -> M b"""
bind(x::Maybe{T}, f::Function) where T = begin
    x isa Nothing && return x 
    return f(x)
end

"""Monad bind: M [a] -> (a -> b) -> M [b]"""
broadcast_bind(x::Maybe{Vector{T}}, f::Function) where T = bind(x, x->map(f, x))

"""f: a -> Int -> b 
wrapping it into func: [a] -> Int -> [b]"""
indexhook_bind(x::Maybe{Vector{T}}, f::Function) where T = begin 
    func(x) = map(1:len(x)) do i 
        f(x[i], i)
    end 
    return bind(x, func)
end


"""
Note that the chain function defined here is not the Haskell's >> function in its Monad definition.
"""

"""Tolerable chain: M [a] -> M [b] -> ([a] -> [b] -> c) -> M c"""
chain(x::Maybe{T1}, y::Maybe{T2}, f::Function, ::Val{:pass}) where {T1, T2} = begin 
    x isa Nothing && return y 
    y isa Nothing && return x 
    return f(x, y)
end

"""blocking chain: M [a] -> M [b] -> ([a] -> [b] -> c) -> M c"""
chain(x::Maybe{T1}, y::Maybe{T2}, f::Function, ::Val{:block}) where {T1, T2} = begin 
    x isa Nothing && return x
    y isa Nothing && return y
    return f(x, y)
end


broadcast_chain(x::Maybe{Vector{T1}}, y::Maybe{Vector{T2}}, f::Function, ::Val{:pass}) where {T1, T2} = begin
    x isa Nothing && return y 
    y isa Nothing && return x 
    @assert len(x)==len(y)
    return chain(x, y, (x,y)->map(t->f(t...), zip(x, y)), Val(:pass))
end

broadcast_chain(x::Maybe{Vector{T1}}, y::Maybe{Vector{T2}}, f::Function, ::Val{:block}) where {T1, T2} = begin
    x isa Nothing && return Nothing 
    y isa Nothing && return Nothing
    @assert len(x)==len(y)
    return chain(x, y, (x,y)->map(t->f(t...), zip(x, y)), Val(:block))
end


######## utils for Vector ########



binary2maybe(binary_pred::Function) = begin 
    func(x) = begin 
        binary_pred(x) && return [x] 
        return nothing 
    end
    return func
end

"""pred: a-> M [a]"""
filterl(pred::Function, vs::Vector{T}) where T = begin
    maybe_pred = binary2maybe(pred) 
    return foldl((x,y)->chain(x, maybe_pred(y), ++, Val(:pass)), vs; init=T[])
end

filterl(pred::Vector{Bool}, vs::Vector{T}) where T = begin
    binaryZip2maybe(y::Tuple{Bool, T}) = begin 
        b, v = y 
        b && return [v] 
        return nothing
    end
    @assert len(pred)==len(vs)
    nvs = collect(zip(pred, vs))
    return foldl((x,y)->chain(x, binaryZip2maybe(y), ++, Val(:pass)), nvs; init=T[])
end