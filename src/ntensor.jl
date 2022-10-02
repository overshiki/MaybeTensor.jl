using MLStyle

abstract type MaybeTensor end
# abstract type MaybeLeafTensor <: MaybeTensor end

(++)(xlist::Vector{T}, ylist::Vector{T}) where T = begin 
    clist = T[]
    append!(clist, xlist)
    append!(clist, ylist)
    return clist
end 

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
# indexhook_bind(x::Union{Vector{T}, Nothing}, f::Function) where T = begin 
#     func(x) = map(1:len(x)) do i 
#         f(x[i], i)
#     end 
#     return bind(x, func)
# end

broadcast_chain(x::Union{Vector{T1}, Nothing}, y::Union{Vector{T2}, Nothing}, f::Function) where T1 where T2 = begin
    x isa Nothing && return y 
    y isa Nothing && return x 
    @assert len(x)==len(y)
    return chain(x, y, (x,y)->map(t->f(t...), zip(x, y)))
end

values(mt::MaybeTensor) = mt.values
(++)(ar::T, br::T) where T<:MaybeTensor = begin
    return T(chain(values(ar), values(br), ++))
end
concat(alist::Vector{T}) where T<:MaybeTensor = foldl(++, alist; init=T(nothing))
toArray(mt::MaybeTensor) = map(toArray, values(mt))

len(mt::MaybeTensor) = len(values(mt))
len(vs::Vector{T}) where T = length(vs)
len(n::Nothing) = 0

struct NSTensor{T<:MaybeTensor}<:MaybeTensor
    values::Union{Vector{T}, Nothing}
end

struct MaybeLeafTensor{T}<:MaybeTensor where T 
    values::Union{Vector{T}, Nothing}
end

const MaybeRealTensor = MaybeLeafTensor{Real}

toArray(mt::MaybeRealTensor) = values(mt)
element(mt::MaybeLeafTensor{T}) where T = begin
    f(x::Vector{T}) = begin 
        @assert len(x)==1
        return x[1]
    end
    return bind(values(mt), f)
end



select(v::Vector{T}, index::Int) where T = Vector{T}([v[index]])
select(v::Nothing, index::Int) = nothing

select(mt::T, index::Int) where T<:MaybeTensor = begin 
    return T(select(values(mt), index))
end

select_element(v::Vector{T}, index::Int) where T = bind(select(v, index), xs->xs[1])
select_value(ns::MaybeTensor, index::Int) = select_element(values(ns), index)

prob_value(ns::NSTensor) = select_value(ns, 1)

len(ns::NSTensor, dimIndex::Int)::Int = begin 
    dimIndex==1 && return len(ns)    
    return len(select_element(values(ns), 1), dimIndex-1)
end

len(ns, dimIndex) = @match (ns, dimIndex) begin 
    (ns::MaybeLeafTensor, 1) => len(ns);
end

size!(ns::MaybeTensor, svec::Vector{Int}) = begin 
    push!(svec, len(ns))
    ns isa MaybeLeafTensor && return svec
    size!(prob_value(ns), svec)
end
size(ns::MaybeTensor)::Tuple = Tuple(size!(ns, Int[]))
dim(ns::MaybeTensor)::Int = length(size(ns))

transpose(ns::NSTensor{NSTensor{T}}) where T = begin 
    inner_len = len(ns, 2)
    map(1:inner_len) do i 
        map(1:len(ns)) do j 
            select(select_value(ns, j), i)
        end |> concat
    end |> NSTensor
end

transpose(ns::NSTensor, startIndex::Int) = begin 
    startIndex==1 && return transpose(ns)
    # map(1:len(ns)) do i 
    #     transpose(select_value(ns, i), startIndex-1)
    # end |> NSTensor 

    return broadcast_bind(ns, x->transpose(x, startIndex-1))
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

rightConstuctor(v::Vector{T}, T2) where T = begin
    T<:MaybeTensor && return NSTensor{T}(v)
    T<:Real && return MaybeLeafTensor{Real}(v)
    T<:Function && return MaybeLeafTensor{Function}(v)
    return MaybeLeafTensor{T}(v)
end
rightConstuctor(v::Nothing, T) = begin 
    return T(v)
end

"""Monad bind: Ma, (a->Mb) -> Mb"""
broadcast_bind(nsa::T, f::Function) where T<:MaybeTensor = begin 
    # return bind(broadcast_bind(values(nsa), f), rightConstuctor)
    return rightConstuctor(broadcast_bind(values(nsa), f), T)
end

# indexhook_bind(nsa::NSTensor, f::Function) = begin 
#     # v = map(1:len(nsa)) do i 
#     #     f(select_value(nsa, i), i)
#     # end
#     # if len(v)==0
#     #     v = nothing 
#     # end
#     # return NSTensor(v) 
#     indexhook_bind(values(nsa), f) |> NSTensor
# end


"""bind function only supports MaybeLeafTensor, since f apply to the whole vector.
For NSTensor, there are element_wise versions called `broadcast_bind` and `hook_bind`
"""
bind(mrt::MaybeLeafTensor{T}, f::Function) where T = MaybeLeafTensor{T}(bind(values(mrt), f))

"""chain function only supports MaybeLeafTensor, since f apply to the whole vector
For NSTensor, there is an element_wise version called `pairchain` 
"""
chain(msa::MaybeLeafTensor{T}, msb::MaybeLeafTensor{T}, f::Function) where T = begin
    return MaybeLeafTensor{T}(chain(values(msa), values(msb), f));
end 

broadcast_chain(msa::MaybeLeafTensor{T}, msb::MaybeLeafTensor{T}, f::Function) where T = begin
    return MaybeLeafTensor{T}(broadcast_chain(values(msa), values(msb), f));
end 

"""only for NSTensor and NSTensor, since the bind function works differently for NSTensor and MaybeLeafTensor, the version fo MaybeLeafTensor is called `chain`"""
# pairchain(nsa::NSTensor, nsb::NSTensor, op::Function) = begin 
#     @assert len(nsa)==len(nsb)
#     return indexhook_bind(nsa, (ns, i)->op(ns, select_value(nsb, i)))
#     # return broadcast_bind(nsa, (ns, i)->op(ns, select_value(nsb, i)))
# end
broadcast_chain(nsa::NSTensor, nsb::NSTensor, op::Function) = begin 
    @assert len(nsa)==len(nsb)
    # return indexhook_bind(nsa, (ns, i)->op(ns, select_value(nsb, i)))
    # return broadcast_bind(nsa, (ns, i)->op(ns, select_value(nsb, i)))
    return broadcast_chain(values(nsa), values(nsb), op) |> NSTensor
end

const fReal = Union{Real, Function}
const ApplyElement      = Tuple{Function, NSTensor, MaybeLeafTensor}
const ApplySkip         = Tuple{Function, NSTensor, fReal}
const ApplySelectValue  = Tuple{Function, NSTensor, NSTensor}
const ApplyBindOp       = Tuple{Function, MaybeLeafTensor, fReal}
const ApplyChainOp      = Tuple{Function, MaybeLeafTensor, MaybeLeafTensor}

op_apply(a, b, c) = @match (a, b, c) begin
    # (op, nsa, mrb)::ApplyElement     => indexhook_bind(nsa, (ns, i)->op(ns, element(mrb)));
    (op, nsa, mrb)::ApplyElement     => broadcast_bind(nsa, ns->op(ns, element(mrb)));
    # (op, nsa, b)::ApplySkip          => indexhook_bind(nsa, (ns, i)->op(ns, b));
    (op, nsa, b)::ApplySkip          => broadcast_bind(nsa, ns->op(ns, b));
    # (bop, nsa, b)::ApplyBindOp       => bind(nsa, x->bop(x, b));
    (bop, nsa, b)::ApplyBindOp       => broadcast_bind(nsa, x->bop(x, b));
    # (bop, nsa, nsb)::ApplyChainOp    => chain(nsa, nsb, bop);
    (bop, nsa, nsb)::ApplyChainOp    => broadcast_chain(nsa, nsb, bop);
    # (op, nsa, nsb)::ApplySelectValue => bind(nsa, (ns, i)->op(ns, select_value(nsb, i)))
    # (op, nsa, nsb)::ApplySelectValue => pairchain(nsa, nsb, op)
    (op, nsa, nsb)::ApplySelectValue => broadcast_chain(nsa, nsb, op)
end


const LowLevelApplyOp = Tuple{MaybeRealTensor, Union{MaybeRealTensor, 
                                                     Real}}

const ApplyOp = Tuple{NSTensor, Union{MaybeRealTensor, 
                                      Real, 
                                      NSTensor}}

const ToPermute = Union{Tuple{Real, Union{NSTensor, 
                                          MaybeRealTensor}}, 
                        Tuple{MaybeRealTensor, NSTensor}}


Base.:*(a, b) = @match (a, b) begin 
    (nsa, b)::Union{LowLevelApplyOp, ApplyOp} => op_apply(*, nsa, b);
    # (nsa, b)::ApplyOp         => op_apply(nproduct, nsa, b);
    # (nsa, b)::ApplyOp         => op_apply(*, nsa, b);
    # (b, nsa)::ToPermute       => nproduct(nsa, b);
    (b, nsa)::ToPermute       => nsa * b;
    (nsa::MaybeRealTensor, b::Nothing) => nsa;
    (b::Nothing, nsa::MaybeRealTensor) => nsa;
end


Base.:+(a, b) = @match (a, b) begin 
    (nsa, b)::Union{LowLevelApplyOp, ApplyOp} => op_apply(+, nsa, b);
    # (nsa, b)::ApplyOp         => op_apply(+, nsa, b);
    (b, nsa)::ToPermute       => nsa + b;
    # (nsa, b)::ApplyOp         => op_apply(nsum, nsa, b);
    # (b, nsa)::ToPermute       => nsum(nsa, b);
    (nsa::MaybeRealTensor, b::Nothing) => nsa;
    (b::Nothing, nsa::MaybeRealTensor) => nsa;
end


Empty(::MaybeLeafTensor{T}) where T = MaybeLeafTensor{T}(nothing)

# Empty(sv::NSTensor)::NSTensor = begin 
#     return NSTensor(map(x->MaybeRealTensor(nothing), 1:len(sv)))
# end
Empty(sv::NSTensor)::NSTensor = broadcast_bind(sv, Empty)
    # map(1:len(sv)) do i 
    #     Empty(select_value(sv, i))
    # end |> NSTensor


# Base.reduce(ob::Function, nsa::NSTensor) = foldl(ob, values(nsa); init=Empty(prob_value(nsa)))
Base.reduce(ob::Function, nsa::NSTensor; init=Empty(prob_value(nsa))) = foldl(ob, values(nsa); init=init)
# sum_reduce(nsa::NSTensor) = Base.reduce(nsum, nsa)
sum_reduce(nsa::NSTensor) = Base.reduce(+, nsa)

# sum_product(nsa::NSTensor, nsb::NSTensor) = sum_reduce(nproduct(nsa, nsb))
sum_product(nsa::NSTensor, nsb::NSTensor) = sum_reduce(nsa * nsb)

prepare_einsum(nsa::NSTensor, insa::Int) = begin 
    schedule = collect(1:dim(nsa))
    target = ([insa] ++ schedule[1:insa-1]) ++ schedule[insa+1:end]
    nsa = transpose(nsa, target)
    return nsa
end
einsum(nsa::NSTensor, nsb::NSTensor, insa::Int, insb::Int)::NSTensor = begin 
    return sum_product(prepare_einsum(nsa, insa), prepare_einsum(nsb, insb))
end

"""Monad chain: Ma, Ma, ((Mb, Mb)->Mc) -> Md, this is for tensor product"""
tpchain(nsa::MaybeTensor, nsb::MaybeTensor, f::Function, innerCon::Type{<:MaybeTensor}) = begin
    # map(1:len(nsa)) do i 
    #     map(1:len(nsb)) do j 
    #         f(select_value(nsa, i), select_value(nsb, j))
    #     end |> innerCon
    # end |> NSTensor

    return broadcast_bind(nsa, x->broadcast_bind(nsb, y->f(x,y)))

end

tensorproduct(nsa::NSTensor, nsb::NSTensor)::NSTensor = tpchain(nsa, nsb, tensorproduct, NSTensor)
# tensorproduct(nsa::NSTensor, msb::MaybeRealTensor) = tpchain(nsa, msb, nproduct, NSTensor)
# tensorproduct(msb::MaybeRealTensor, nsa::NSTensor) = tpchain(nsa, msb, nproduct, NSTensor) |> transpose
tensorproduct(nsa::NSTensor, msb::MaybeRealTensor) = tpchain(nsa, msb, *, NSTensor)
tensorproduct(msb::MaybeRealTensor, nsa::NSTensor) = tpchain(nsa, msb, *, NSTensor) |> transpose
tensorproduct(msa::MaybeRealTensor, msb::MaybeRealTensor) = tpchain(msa, msb, .*, MaybeRealTensor)


default_NSTensor(num::Int) = begin 
    r_vec = map(x->MaybeRealTensor([x]), 1:num)
    return NSTensor(r_vec)
end

default_NSTensor(ns::NSTensor, num::Int) = begin
    r_vec = map(x->ns, 1:num)
    return NSTensor(r_vec)
end

disArray(x::MaybeTensor) = x |> toArray |> display


