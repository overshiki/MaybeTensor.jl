
abstract type MaybeTensor end

struct State 
    id::Symbol
end
State(x::Union{Int, String}) = State(Symbol(x))
(++)(xlist::Vector{T}, ylist::Vector{T}) where T = begin 
    clist = T[]
    append!(clist, xlist)
    append!(clist, ylist)
    return clist
end 

"""Monad chain: ((Ma, Ma), (Ma, Ma)-> Ma) -> Ma"""
chain(x::Union{Vector{T}, Nothing}, y::Union{Vector{T}, Nothing}, f::Function) where T = begin 
    x isa Nothing && return y 
    y isa Nothing && return x 
    return f(x, y)
end
"""Monad bind: Ma, (a->Mb) -> Mb"""
bind(x::Union{Vector{T}, Nothing}, f::Function) where T = begin
    x isa Nothing && return x 
    return f(x)
end

states(mt::MaybeTensor) = mt.states 
values(mt::MaybeTensor) = mt.values
(++)(ar::T, br::T) where T<:MaybeTensor = T(chain(states(ar), states(br), ++), chain(values(ar), values(br), ++))
concat(alist::Vector{T}) where T<:MaybeTensor = foldl(++, alist; init=T(nothing, nothing))
toArray(mt::MaybeTensor) = map(toArray, values(mt))

len(mt::MaybeTensor) = len(values(mt))
len(vs::Vector{T}) where T = length(vs)
len(n::Nothing) = 0


struct MaybeRealTensor<:MaybeTensor
    states::Union{Vector{State}, Nothing}
    values::Union{Vector{Real}, Nothing}
end
toArray(mt::MaybeRealTensor) = values(mt)


struct NSTensor{T<:MaybeTensor}<:MaybeTensor
    states::Union{Vector{State}, Nothing}
    values::Union{Vector{T}, Nothing}
end

select(v::Vector{T}, index::Int) where T = Vector{T}([v[index]])
select(v::Nothing, index::Int) = nothing

select(mt::T, index::Int) where T<:MaybeTensor = begin 
    return T(select(states(mt), index), select(values(mt), index))
end

select_element(v::Vector{T}, index::Int) where T = bind(select(v, index), xs->xs[1])

len(ns::NSTensor, dimIndex::Int)::Int = begin 
    dimIndex==1 && return len(ns)    
    return len(select_element(values(ns), 1), dimIndex-1)
end

transpose(ns::NSTensor{NSTensor{T}}) where T = begin 
    inner_len = len(ns, 2)
    v = map(1:inner_len) do i 
        return map(1:len(ns)) do j 
            return select(select_element(values(ns), j), i)
        end |> concat
    end
    return NSTensor(states(select_element(values(ns), 1)), v)
end


state_vec = map(State, 1:3)
r_vec = map(x->MaybeRealTensor(nothing, [x]), 1:3)
ns = NSTensor(state_vec, r_vec)
nns = NSTensor(map(State, 1:2), [ns, ns])

nns ++ nns
concat([nns, nns])
typeof(nns)
toArray(nns) |> display
# len(nns)
toArray(transpose(nns)) |> display

@show len(nns, 1), len(nns, 2)