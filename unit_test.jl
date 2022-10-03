include("src/ntensor.jl")



default_test() = begin 

    # state_vec = map(State, 1:3)
    # r_vec = map(x->MaybeRealTensor(nothing, [x]), 1:3)
    # ns = NSTensor(state_vec, r_vec)
    # nns = NSTensor(map(State, 1:2), [ns, ns])

    ns = default_NSTensor(3)
    nns = default_NSTensor(ns, 2)


    nns ++ nns
    concat([nns, nns])
    typeof(nns)
    toNSVector(nns) |> display
    # len(nns)
    toNSVector(transpose(nns)) |> display

    @show len(nns, 1), len(nns, 2)

    # nnns = NSTensor(map(State, 1:2), [nns, nns])
    nnns = default_NSTensor(nns, 2)
    @show size(nnns)
    tnnns = transpose(nnns, 2)
    @show size(tnnns)

    # @show transpose_schedule([3,2,1])
    # @show transpose_schedule([2, 3, 1])
    tnnns = transpose(nnns, [3,2,1])
    @show size(tnnns)

    nns |> disArray
    # nproduct(nns, nns) |> disArray

    # nproduct(nns, 3) |> disArray
    # nproduct(nns, MaybeRealTensor([3])) |> disArray
    nns * nns |> disArray

    nns * 3 |> disArray
    nns * MaybeRealTensor([3]) |> disArray

    # nsum(nns, MaybeRealTensor([3])) |> disArray
    nns + MaybeRealTensor([3]) |> disArray


    sum_reduce(nnns) |> disArray
    # reduce(nsum, nnns) |> disArray
    reduce(+, nnns) |> disArray

    sum_reduce(nns) |> disArray
    sum_reduce(nns) |> sum_reduce |> disArray

    sum_reduce(nnns) |> sum_reduce |> disArray
    sum_reduce(nnns) |> sum_reduce |> sum_reduce |> disArray

    nns |> disArray
    sum_product(nns, nns) |> disArray
end

test_sum_product() = begin 
    nsa = default_NSTensor(3) |> x -> default_NSTensor(x, 2)
    nsb = default_NSTensor(2)
    disArray(nsa)
    disArray(nsb)
    @show size(nsa), size(nsb)

    ns = sum_product(nsa, nsb)
    disArray(ns)
    @show size(ns)
end

exam() = begin 
    v = [1,2] |> x -> reshape(x, (1, 2))
    m = [1 2 3; 1 2 3]
    v |> display
    m |> display
    v * m |> display
end

test_einsum() = begin 
    nsa = default_NSTensor(3) |> x -> default_NSTensor(x, 2) |> transpose
    nsb = default_NSTensor(2)
    disArray(nsa)
    disArray(nsb)
    @show size(nsa), size(nsb)

    ns = einsum(nsa, nsb, 2, 1)
    disArray(ns)
    @show size(ns)
end

test_tensorproduct() = begin 
    nsa = default_NSTensor(3) |> x -> default_NSTensor(x, 2) |> transpose
    nsb = default_NSTensor(2)
    disArray(nsa)
    disArray(nsb)
    @show size(nsa), size(nsb)

    tp = tensorproduct(nsa, nsb)
    @show size(tp)

    tp = tensorproduct(nsa, nsa)
    @show size(tp)

    tp = tensorproduct(nsb, nsa)
    @show size(tp)
end

include("src/ftensor.jl")
test_ftensor() = begin 
    default_fNSTensor(num::Int) = begin 
        r_vec = map(x->MaybeFuncTensor([x->x*x]), 1:num)
        return NSTensor(r_vec)
    end

    fnsa = default_fNSTensor(3) |> x -> default_NSTensor(x, 2) |> transpose
    fnsb = default_fNSTensor(2)

    @show size(fnsa), size(fnsb)

    ftp1 = tensorproduct(fnsa, fnsb)
    @show size(ftp1)

    tp = tensorproduct(fnsa, fnsa)
    @show size(tp)

    tp = tensorproduct(fnsb, fnsa)
    @show size(tp)


    nsa = default_NSTensor(3) |> x -> default_NSTensor(x, 2) |> transpose
    nsb = default_NSTensor(2)

    apply(fnsa, nsa) |> disArray
    apply(fnsb, nsb) |> disArray

    tp1 = tensorproduct(nsa, nsb)
    apply(ftp1, tp1) |> disArray

    fnsa = default_fNSTensor(3)
end


test_flatten() = begin 
    nsa = default_NSTensor(3) |> x -> default_NSTensor(x, 2) |> transpose |> x -> default_NSTensor(x, 4)
    @show size(nsa)
    # disArray(nsa)
    fsa = flatten(nsa)
    @show size(fsa)
    # disArray(fsa)    
    rfsa = reshape(fsa, [4, 3, 2])
    @show size(rfsa)
    @show rfsa == nsa
    @show rfsa == fsa 
    # disArray(rfsa)
end

test_empty() = begin
    @show Empty(MaybeRealTensor)
    @show Empty([3, 2], MaybeRealTensor) |> size
end

include("src/sparse.jl")
test_sparse() = begin 

    i = [[1,2,3], [1,2,3]]
    shape = [3,3]
    v = map(x->MaybeRealTensor([x]), 1:3)
    ct = coo_tensor(i, v, shape)
    @show sum_reduce(ct) |> sum_reduce

    i = [[1,1,2,3], [1,2,2,3], [3,3,3,3]]
    @show prune_indices(i, 1)
    shape = [3,3,3]
    v = map(x->MaybeRealTensor([x]), 1:4)
    ct = coo_tensor(i, v, shape)
    @show sum_reduce(ct) |> sum_reduce |> sum_reduce
end

test_NSVector() = begin 
    i = [[1,1,2,3], [1,2,2,3], [3,3,3,3]]
    fromNSVector(i) |> size
    fromNSVector(i) |> transpose |> size
end

test_filterl() = begin 
    a = [1,2,3,4]
    pred = x->x>2
    @show filterl(pred, a)
    pred = [true, false, true, false]
    @show filterl(pred, a)
end

test_filterl()

test_NSVector()

test_sparse()
test_empty()

test_flatten()

default_test()
test_sum_product()
exam()
test_einsum()

test_tensorproduct()

test_ftensor()