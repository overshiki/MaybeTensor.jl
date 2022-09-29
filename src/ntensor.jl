
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
element(mt::MaybeRealTensor) = begin
    f(x::Vector{Real}) = begin 
        @assert len(x)==1
        return x[1]
    end
    return bind(values(mt), f)
end

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
select_value(ns::NSTensor, index::Int) = select_element(values(ns), index)
len(ns::NSTensor, dimIndex::Int)::Int = begin 
    dimIndex==1 && return len(ns)    
    return len(select_element(values(ns), 1), dimIndex-1)
end

size!(ns::MaybeTensor, svec::Vector{Int}) = begin 
    push!(svec, len(ns))
    ns isa MaybeRealTensor && return svec
    size!(select_value(ns, 1), svec)
end
size(ns::NSTensor)::Tuple = Tuple(size!(ns, Int[]))

transpose(ns::NSTensor{NSTensor{T}}) where T = begin 
    inner_len = len(ns, 2)
    v = map(1:inner_len) do i 
        return map(1:len(ns)) do j 
            return select(select_value(ns, j), i)
        end |> concat
    end
    return NSTensor(states(select_value(ns, 1)), v)
end

transpose(ns::NSTensor, startIndex::Int) = begin 
    startIndex==1 && return transpose(ns)
    v = map(1:len(ns)) do i 
        return transpose(select_value(ns, i), startIndex-1)
    end 
    return NSTensor(states(ns), v)
end




struct StartIndex 
    index::Int
end

"""
    (1, 2, 3) -> (2, 3, 1) -> (1, 2), (2, 3)
    (1, 2, 3) -> (3, 1, 2) -> (2, 3), (1, 2)
    (1, 2, 3) -> (3, 2, 1) -> (1, 2), (2, 3), (1, 2)
"""
transpose_schedule(orders::Vector{Int}, schedule::Vector{StartIndex}) = begin
    len(orders)==0 && return schedule
    o, os = orders[end], orders[1:end-1]    
    schedule = schedule ++ map(StartIndex, o:len(orders)-1)

    len(os)==0 && return schedule

    nos = map(os) do oi 
        oi > o && return oi -1 
        oi < o && return oi
    end
    transpose_schedule(nos, schedule)
end
transpose_schedule(orders::Vector{Int}) = transpose_schedule(orders, StartIndex[])


transpose(ns::NSTensor, schedule::Vector{StartIndex})::NSTensor = foldl((ns, s)->transpose(ns, s.index), schedule; init=ns)
transpose(ns::NSTensor, targetVec::Vector{Int})::NSTensor = transpose(ns, transpose_schedule(targetVec))

nproduct(nsa::MaybeRealTensor, nsb::MaybeRealTensor) = MaybeRealTensor(states(nsa), chain(values(nsa), values(nsb), .*))
# nproduct(nsa::NSTensor{T}, nsb::NSTensor{T}) where T = begin 
#     @assert states(nsa)==states(nsb)
#     v = map(1:len(nsa)) do i 
#         return nproduct(select_value(nsa, i), select_value(nsb, i))
#     end
#     return NSTensor(states(nsa), v)
# end

"""Monad bind: Ma, (a->Mb) -> Mb"""
bind(nsa::NSTensor, f::Function) = begin 
    v = map(1:len(nsa)) do i 
        return f(select_value(nsa, i), i)
    end
    return NSTensor(states(nsa), v) 
end

nproduct(nsa::NSTensor, mrb, f::Function) = begin 
    v = map(1:len(nsa)) do i 
        return nproduct(select_value(nsa, i), f(mrb, i))
    end
    return NSTensor(states(nsa), v)
end

nproduct(nsa::NSTensor, mrb::MaybeRealTensor) = begin 
    f(ns, i) = nproduct(ns, element(mrb))
    return bind(nsa, f)
    # v = map(1:len(nsa)) do i 
    #     return nproduct(select_value(nsa, i), element(mrb))
    # end
    # return NSTensor(states(nsa), v)
    # return nproduct(nsa, mrb, (mrb, i)->element(mrb))
end
nproduct(mrb::MaybeRealTensor, nsa::NSTensor) = nproduct(nsa, mrb)


nproduct(nsa::NSTensor, b::Real) = begin 
    f(ns, i) = nproduct(ns, b)
    return bind(nsa, f)
    # v = map(1:len(nsa)) do i 
    #     return nproduct(select_value(nsa, i), b)
    # end
    # return NSTensor(states(nsa), v)
    # return nproduct(nsa, b, (x, i)->x)
end
nproduct(b::Real, nsa::NSTensor) = nproduct(nsa, b)

nproduct(nsa::MaybeRealTensor, b::Real) = MaybeRealTensor(states(nsa), bind(values(nsa), x->x.*b))
nproduct(b::Real, nsa::MaybeRealTensor) = nproduct(nsa, b)

nproduct(nsa::NSTensor, nsb::NSTensor) = begin 
    @assert states(nsa)==states(nsb)
    f(ns, i) = nproduct(ns, select_value(nsb, i))
    return bind(nsa, f)

    # v = map(1:len(nsa)) do i 
    #     return nproduct(select_value(nsa, i), select_value(nsb, i))
    # end
    # return NSTensor(states(nsa), v)
    # return nproduct(nsa, nsb, (nsb, i)->select_value(nsb, i))
end


sum_product(nsa::NSTensor, nsb::NSTensor) = begin 

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

nnns = NSTensor(map(State, 1:2), [nns, nns])
@show size(nnns)
tnnns = transpose(nnns, 2)
@show size(tnnns)

# @show transpose_schedule([3,2,1])
# @show transpose_schedule([2, 3, 1])
tnnns = transpose(nnns, [3,2,1])
@show size(tnnns)

nns |> toArray |> display
nproduct(nns, nns) |> toArray |> display

nproduct(nns, 3) |> toArray |> display
nproduct(nns, MaybeRealTensor(nothing, [3])) |> toArray |> display
