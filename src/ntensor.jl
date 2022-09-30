using MLStyle

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
(++)(ar::T, br::T) where T<:MaybeTensor = begin
    return T(chain(states(ar), states(br), ++), chain(values(ar), values(br), ++))
end
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
prob_value(ns::NSTensor) = select_value(ns, 1)

len(ns::NSTensor, dimIndex::Int)::Int = begin 
    dimIndex==1 && return len(ns)    
    return len(select_element(values(ns), 1), dimIndex-1)
end

len(ns, dimIndex) = @match (ns, dimIndex) begin 
    (ns::MaybeRealTensor, 1) => len(ns);
end

size!(ns::MaybeTensor, svec::Vector{Int}) = begin 
    push!(svec, len(ns))
    ns isa MaybeRealTensor && return svec
    size!(prob_value(ns), svec)
end
size(ns::MaybeTensor)::Tuple = Tuple(size!(ns, Int[]))

transpose(ns::NSTensor{NSTensor{T}}) where T = begin 
    inner_len = len(ns, 2)
    v = map(1:inner_len) do i 
        return map(1:len(ns)) do j 
            return select(select_value(ns, j), i)
        end |> concat
    end
    return NSTensor(states(prob_value(ns)), v)
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


transpose(ns::NSTensor, schedule::Vector{StartIndex})::NSTensor = begin
    return foldl((ns, s)->transpose(ns, s.index), schedule; init=ns)
end
transpose(ns::NSTensor, targetVec::Vector{Int})::NSTensor = begin
    return transpose(ns, transpose_schedule(targetVec))
end


"""Monad bind: Ma, (a->Mb) -> Mb"""
bind(nsa::NSTensor, f::Function) = begin 
    v = map(1:len(nsa)) do i 
        return f(select_value(nsa, i), i)
    end
    if len(v)==0
        v = nothing 
    end
    return NSTensor(states(nsa), v) 
end
bind(mrt::MaybeRealTensor, f::Function) = MaybeRealTensor(states(mrt), bind(values(mrt), f))
chain(msa::MaybeRealTensor, msb::MaybeRealTensor, f::Function) = begin
    return MaybeRealTensor(states(msa), chain(values(msa), values(msb), f));
end 

const ApplyElement      = Tuple{Function, NSTensor, MaybeRealTensor}
const ApplySkip         = Tuple{Function, NSTensor, Real}
const ApplySelectValue  = Tuple{Function, NSTensor, NSTensor}
const ApplyBindOp       = Tuple{Function, MaybeRealTensor, Real}
const ApplyChainOp      = Tuple{Function, MaybeRealTensor, MaybeRealTensor}

op_apply(a, b, c) = @match (a, b, c) begin
    (op, nsa, mrb)::ApplyElement     => bind(nsa, (ns, i)->op(ns, element(mrb)));
    (op, nsa, b)::ApplySkip          => bind(nsa, (ns, i)->op(ns, b));
    (bop, nsa, b)::ApplyBindOp       => bind(nsa, x->bop(x, b));
    (bop, nsa, nsb)::ApplyChainOp    => chain(nsa, nsb, bop);
    (op, nsa, nsb)::ApplySelectValue => begin 
                                            @assert states(nsa)==states(nsb)
                                            return bind(nsa, (ns, i)->op(ns, select_value(nsb, i)))
                                        end
end


const LowLevelApplyOp = Tuple{MaybeRealTensor, Union{MaybeRealTensor, 
                                                     Real}}

const ApplyOp = Tuple{NSTensor, Union{MaybeRealTensor, 
                                      Real, 
                                      NSTensor}}

const ToPermute = Union{Tuple{Real, Union{NSTensor, 
                                          MaybeRealTensor}}, 
                        Tuple{MaybeRealTensor, NSTensor}}


nproduct(a, b) = @match (a, b) begin 
    (nsa, b)::LowLevelApplyOp => op_apply(.*, nsa, b);
    (nsa, b)::ApplyOp         => op_apply(nproduct, nsa, b);
    (b, nsa)::ToPermute       => nproduct(nsa, b);
    (nsa::MaybeRealTensor, b::Nothing) => nsa;
    (b::Nothing, nsa::MaybeRealTensor) => nsa;
end


nsum(a, b) = @match (a, b) begin 
    (nsa, b)::LowLevelApplyOp => op_apply(.+, nsa, b);
    (nsa, b)::ApplyOp         => op_apply(nsum, nsa, b);
    (b, nsa)::ToPermute       => nsum(nsa, b);
    (nsa::MaybeRealTensor, b::Nothing) => nsa;
    (b::Nothing, nsa::MaybeRealTensor) => nsa;
end


Empty(::MaybeRealTensor) = MaybeRealTensor(nothing, nothing)
Empty(sv::NSTensor)::NSTensor = begin 
    return NSTensor(states(sv), map(x->MaybeRealTensor(nothing, nothing), 1:len(sv)))
end

nreduce(nsa::NSTensor, ob::Function) = foldl(ob, values(nsa); init=Empty(prob_value(nsa)))
sum_reduce(nsa::NSTensor) = nreduce(nsa, nsum)

sum_product(nsa::NSTensor, nsb::NSTensor) = sum_reduce(nproduct(nsa, nsb))


default_NSTensor(num::Int) = begin 
    state_vec = map(State, 1:num)
    r_vec = map(x->MaybeRealTensor(nothing, [x]), 1:num)
    return NSTensor(state_vec, r_vec)
end

default_NSTensor(ns::NSTensor, num::Int) = begin
    state_vec = map(State, 1:num)
    r_vec = map(x->ns, 1:num)
    return NSTensor(state_vec, r_vec)
end

disArray(x::MaybeTensor) = x |> toArray |> display


