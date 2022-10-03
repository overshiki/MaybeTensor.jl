
const MaybeFuncTensor = MaybeLeafTensor{Function}

func_chain(f1::Function, f2::Function) = x -> x |> f1 |> f2
func_chain(sf1::Vector{Function}, f2::Function) = map(f->func_chain(f, f2), sf1)
func_chain(f1::Function, sf2::Vector{Function}) = map(f->func_chain(f1, f), sf2)


const fLowLevelApplyOp = Tuple{MaybeFuncTensor, Union{MaybeFuncTensor, 
                                                     Function}}

const fApplyOp = Tuple{NSTensor, Union{MaybeFuncTensor, 
                                      Function, 
                                      NSTensor}}

const fToPermute = Union{Tuple{Function, Union{NSTensor, 
                                          MaybeFuncTensor}}, 
                        Tuple{MaybeFuncTensor, NSTensor}}


compose(a, b) = @match (a, b) begin 
    (nsa, b)::fLowLevelApplyOp => op_apply(func_chain, nsa, b);
    (nsa, b)::fApplyOp         => op_apply(compose, nsa, b);
    (b, nsa)::fToPermute       => compose(nsa, b);
    (nsa::MaybeFuncTensor, b::Nothing) => nsa;
    (b::Nothing, nsa::MaybeFuncTensor) => nsa;
end

tensorproduct(nsa::NSTensor, msb::MaybeFuncTensor) = tpchain(nsa, msb, compose)
tensorproduct(msb::MaybeFuncTensor, nsa::NSTensor) = tpchain(nsa, msb, compose) |> transpose
tensorproduct(msa::MaybeFuncTensor, msb::MaybeFuncTensor) = tpchain(msa, msb, func_chain)

apply(nsa::NSTensor, nsb::NSTensor)::NSTensor = begin 
    broadcast_chain(nsa, nsb, apply)
end

apply(nsa::MaybeFuncTensor, nsb::MaybeRealTensor)::MaybeRealTensor = begin 
    @assert len(nsa)==len(nsb)
    map(1:len(nsa)) do i
        func(vnsa, vnsb) = begin 
            vnsb[i] |> vnsa[i]
        end
        return chain(values(nsa), values(nsb), func)
    end |> MaybeRealTensor
end