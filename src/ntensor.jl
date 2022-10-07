using MLStyle
include("utils.jl")

abstract type AbstractMaybeTensor end
# abstract type MaybeLeafTensor <: AbstractMaybeTensor end

######## NSTensor and MaybeLeafTensor ########
struct NSTensor{T<:AbstractMaybeTensor}<:AbstractMaybeTensor
    values::Maybe{Vector{T}}
end

struct MaybeLeafTensor{T}<:AbstractMaybeTensor where T 
    values::Maybe{Vector{T}}
end

######## define MaybeRealTensor ########
const MaybeRealTensor = MaybeLeafTensor{Real}




######## reduce, sum_product and einsum ########
# Empty(::Type{MaybeLeafTensor{T}}) where T = MaybeLeafTensor{T}(nothing)
Empty(::Type{T}) where T<:AbstractMaybeTensor = T(nothing)
Empty(::MaybeLeafTensor{T}) where T = MaybeLeafTensor{T}(nothing)

"""empty tensor with the same shape as sv"""
Empty(sv::NSTensor)::NSTensor = broadcast_bind(sv, Empty)

Empty(v::AbstractMaybeTensor, shape::Vector{Int}) = begin
    len(shape)==0 && return v 
    xs, x = shape[1:end-1], shape[end]
    nv = map(x->Empty(v), 1:x) |> NSTensor
    return Empty(nv, xs)
end

(Empty(shape::Vector{Int}, ::Type{MaybeLeafTensor{T}})::NSTensor) where T = begin 
    xs, x = shape[1:end-1], shape[end]
    v = map(x->Empty(MaybeLeafTensor{T}), 1:x) |> NSTensor
    return Empty(v, xs)    
end

######## utils for AbstractMaybeTensor and Vector ########


values(mt::AbstractMaybeTensor) = mt.values
(++)(ar::T, br::T) where T<:AbstractMaybeTensor = begin
    return T(chain(values(ar), values(br), ++, Val(:pass)))
end
concat(alist::Vector{T}) where T<:AbstractMaybeTensor = foldl(++, alist; init=Empty(T))

######## working with nested vector ########
toNSVector(mt::AbstractMaybeTensor) = map(toNSVector, values(mt))
toNSVector(mt::MaybeLeafTensor{T}) where T<:Real = values(mt)

fromNSVector(nsv::Vector{Vector{T}}) where T = map(fromNSVector, nsv) |> NSTensor
fromNSVector(nsv::Vector{T}) where T<:Real = map(x->MaybeLeafTensor{T}([x]), nsv) |> NSTensor

######## len, element ########
len(mt::AbstractMaybeTensor) = len(values(mt))
len(n::Nothing) = 0


element(mt::MaybeLeafTensor{T}) where T = begin
    f(x::Vector{T}) = begin 
        @assert len(x)==1
        return x[1]
    end
    return bind(values(mt), f)
end


######## select, select_element, select_value, prob_value ########

select(mt::T, index::Int) where T<:AbstractMaybeTensor = begin 
    return T(select(values(mt), index))
end
select_value(ns::AbstractMaybeTensor, index::Int) = select_element(values(ns), index)

prob_value(ns::NSTensor) = select_value(ns, 1)

######## len, size, ndims ########
len(ns::NSTensor, dimIndex::Int)::Int = begin 
    dimIndex==1 && return len(ns)    
    return len(select_element(values(ns), 1), dimIndex-1)
end

len(ns, dimIndex) = @match (ns, dimIndex) begin 
    (ns::MaybeLeafTensor, 1) => len(ns);
end

size!(ns::AbstractMaybeTensor, svec::Vector{Int}) = begin 
    push!(svec, len(ns))
    ns isa MaybeLeafTensor && return svec
    size!(prob_value(ns), svec)
end
Base.size(ns::AbstractMaybeTensor)::Tuple = Tuple(size!(ns, Int[]))
Base.ndims(ns::AbstractMaybeTensor)::Int = length(size(ns))



######## transpose ########
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


######## broadcast_bind for AbstractMaybeTensor ########
rightConstuctor(v::Vector{T}, T2) where T = begin
    T<:AbstractMaybeTensor && return NSTensor{T}(v)
    T<:Real && return MaybeLeafTensor{Real}(v)
    T<:Function && return MaybeLeafTensor{Function}(v)
    return MaybeLeafTensor{T}(v)
end
rightConstuctor(v::Nothing, T) = begin 
    return T(v)
end

"""Monad bind: Ma, (a->Mb) -> Mb"""
broadcast_bind(nsa::T, f::Function) where T<:AbstractMaybeTensor = begin 
    # return bind(broadcast_bind(values(nsa), f), rightConstuctor)
    return rightConstuctor(broadcast_bind(values(nsa), f), T)
end

indexhook_bind(nsa::NSTensor, f::Function) = begin 
    # v = map(1:len(nsa)) do i 
    #     f(select_value(nsa, i), i)
    # end
    # if len(v)==0
    #     v = nothing 
    # end
    # return NSTensor(v) 
    indexhook_bind(values(nsa), f) |> NSTensor
end


"""bind function only supports MaybeLeafTensor, since f apply to the whole vector.
For NSTensor, there are element_wise versions called `broadcast_bind` and `hook_bind`
"""
bind(mrt::MaybeLeafTensor{T}, f::Function) where T = MaybeLeafTensor{T}(bind(values(mrt), f))

const PassOrBlock = Union{Val{:pass}, Val{:block}}

"""chain function only supports MaybeLeafTensor, since f apply to the whole vector
For NSTensor, there is an element_wise version called `pairchain` 
"""
chain(msa::MaybeLeafTensor{T}, msb::MaybeLeafTensor{T}, f::Function, val::PassOrBlock) where T = begin
    return MaybeLeafTensor{T}(chain(values(msa), values(msb), f, val));
end 

######## broadcast operators ########
broadcast_chain(msa::MaybeLeafTensor{T}, msb::MaybeLeafTensor{T}, f::Function, val::PassOrBlock) where T = begin
    return MaybeLeafTensor{T}(broadcast_chain(values(msa), values(msb), f, val));
end 

# broadcast_chain(msa::MaybeLeafTensor{T}, msb::MaybeLeafTensor{T}, f::Function, ::Val{:block}) where T = begin
#     return MaybeLeafTensor{T}(broadcast_chain(values(msa), values(msb), f, Val(:block)));
# end 

"""only for NSTensor and NSTensor, since the bind function works differently for NSTensor and MaybeLeafTensor, the version fo MaybeLeafTensor is called `chain`"""
broadcast_chain(nsa::NSTensor, nsb::NSTensor, op::Function, val::PassOrBlock) = begin 
    @assert len(nsa)==len(nsb)
    return broadcast_chain(values(nsa), values(nsb), op, val) |> NSTensor
end

# broadcast_chain(nsa::NSTensor, nsb::NSTensor, op::Function, ::Val{:block}) = begin 
#     @assert len(nsa)==len(nsb)
#     return broadcast_chain(values(nsa), values(nsb), op, Val(:block)) |> NSTensor
# end

const fReal = Union{Real, Function}
const ApplyElement      = Tuple{Function, NSTensor, MaybeLeafTensor}
const ApplySkip         = Tuple{Function, NSTensor, fReal}
const ApplySelectValue  = Tuple{Function, NSTensor, NSTensor}
const ApplyBindOp       = Tuple{Function, MaybeLeafTensor, fReal}
const ApplyChainOp      = Tuple{Function, MaybeLeafTensor, MaybeLeafTensor}


op_apply((a, b, c), val) = @match ((a, b, c), val) begin
    ((op, nsa, mrb), val)::Tuple{ApplyElement, PassOrBlock}     => broadcast_bind(nsa, ns->op(ns, element(mrb)));
    ((op, nsa, b), val)::Tuple{ApplySkip, PassOrBlock}          => broadcast_bind(nsa, ns->op(ns, b));
    ((bop, nsa, b), val)::Tuple{ApplyBindOp, PassOrBlock}       => broadcast_bind(nsa, x->bop(x, b));
    ((bop, nsa, nsb), val)::Tuple{ApplyChainOp, PassOrBlock}    => broadcast_chain(nsa, nsb, bop, val);
    ((op, nsa, nsb), val)::Tuple{ApplySelectValue, PassOrBlock} => broadcast_chain(nsa, nsb, op, val)
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
    (nsa, b)::Union{LowLevelApplyOp, ApplyOp} => op_apply((*, nsa, b), Val(:block));
    (b, nsa)::ToPermute       => nsa * b;
    (nsa::MaybeRealTensor, b::Nothing) => nsa;
    (b::Nothing, nsa::MaybeRealTensor) => nsa;
end


Base.:+(a, b) = @match (a, b) begin 
    (nsa, b)::Union{LowLevelApplyOp, ApplyOp} => op_apply((+, nsa, b), Val(:pass));
    (b, nsa)::ToPermute       => nsa + b;
    (nsa::MaybeRealTensor, b::Nothing) => nsa;
    (b::Nothing, nsa::MaybeRealTensor) => nsa;
end



Base.reduce(ob::Function, nsa::NSTensor; init=Empty(prob_value(nsa))) = foldl(ob, values(nsa); init=init)
sum_reduce(nsa::NSTensor) = Base.reduce(+, nsa)

sum_product(nsa::NSTensor, nsb::NSTensor) = sum_reduce(nsa * nsb)

prepare_einsum(nsa::NSTensor, insa::Int) = begin 
    schedule = collect(1:ndims(nsa))
    target = ([insa] ++ schedule[1:insa-1]) ++ schedule[insa+1:end]
    nsa = transpose(nsa, target)
    return nsa
end
einsum(nsa::NSTensor, nsb::NSTensor, insa::Int, insb::Int)::NSTensor = begin 
    return sum_product(prepare_einsum(nsa, insa), prepare_einsum(nsb, insb))
end

######## tensor product ########
"""Monad chain: Ma, Ma, ((Mb, Mb)->Mc) -> Md, this is for tensor product"""
tpchain(nsa::AbstractMaybeTensor, nsb::AbstractMaybeTensor, f::Function) = broadcast_bind(nsa, 
                                                        x->broadcast_bind(nsb, 
                                                        y->f(x,y)))


tensorproduct(nsa::NSTensor, nsb::NSTensor)::NSTensor = tpchain(nsa, nsb, tensorproduct)
tensorproduct(nsa::NSTensor, msb::MaybeRealTensor) = tpchain(nsa, msb, *)
tensorproduct(msb::MaybeRealTensor, nsa::NSTensor) = tpchain(nsa, msb, *) |> transpose
tensorproduct(msa::MaybeRealTensor, msb::MaybeRealTensor) = tpchain(msa, msb, *)

######## flatten and reshape ########
_flatten(ns::NSTensor) = begin 
    _flatten(broadcast_bind(values(ns), _flatten))
end
_flatten(vs::Vector{Vector{T}}) where T = foldl(++, vs; init=T[])
_flatten(vs::MaybeLeafTensor) = [vs]
flatten(ns::NSTensor{T}) where T = _flatten(ns) |> x->rightConstuctor(x, T)

(splitstack(store::Vector{Vector{T}}, vs::Vector{T}, slen::Int)::Vector{Vector{T}}) where T = begin 
    len(vs)==0 && return store
    s, vs = vs[1:slen], vs[slen+1:end]
    return splitstack(store++[s], vs, slen)
end
(Base.reshape(ns::NSTensor{T}, shape::Tuple{Int, Int})::NSTensor{NSTensor{T}}) where T<:MaybeLeafTensor = begin 
    xlen, slen = shape
    nns = splitstack(Vector{T}[], values(ns), slen)
    @assert len(nns)==xlen
    nns = map(NSTensor, nns) |> NSTensor 
    return nns
end
Base.reshape(ns::NSTensor, shape::Vector{Int})::NSTensor = begin
    len(shape)<=1 && return ns
    xlen, sshape = shape[1], shape[2:end]
    slen = foldl(*, sshape; init=1)
    nns = reshape(ns, (xlen, slen))
    return broadcast_bind(nns, x->reshape(x, sshape))
end

######## compare operators ########
Base.:(==)(nsa::AbstractMaybeTensor, nsb::AbstractMaybeTensor)::Bool = begin 
    len(nsa)!==len(nsb) && return false
    broadcast_chain(values(nsa), values(nsb), ==, Val(:pass)) |> all
end


######## default constructor ########
default_NSTensor(num::Int)::NSTensor = begin 
    r_vec = map(x->MaybeRealTensor([x]), 1:num)
    return NSTensor(r_vec)
end

default_NSTensor(ns::NSTensor, num::Int)::NSTensor = begin
    r_vec = map(x->ns, 1:num)
    return NSTensor(r_vec)
end

disArray(x::AbstractMaybeTensor) = x |> toNSVector |> fullstack |> cleanArray |> display
#     dims = dim(x)
#     @show dims, size(x)
#     @assert size(x)[end] == 1
#     x = x |> toNSVector 
#     @show size(x), x
#     error()
#     # |> x->dropdims(x;dims=dims) |> display
# end


