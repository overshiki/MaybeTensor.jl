include("ntensor.jl")

rm_strSpace(str::String)::String = begin 
    length(str)==0 && return str
    str[1]==' ' && return rm_strSpace(str[2:end])
    str[end]==' ' && return rm_strSpace(str[1:end-1])
    return str
end


abstract type StringParser end
const SuStringParser = Union{String, StringParser}

struct Clause <: StringParser
    left::Char 
    right::Char
    content::Maybe{SuStringParser}
end
Clause(cl::Clause, content::Maybe{SuStringParser}) = Clause(cl.left, cl.right, content)

struct SepTwo <: StringParser 
    sep::String
    left::Maybe{SuStringParser}
    right::Maybe{SuStringParser}
end
SepTwo(st::SepTwo, left::Maybe{SuStringParser}, right::Maybe{SuStringParser}) = SepTwo(st.sep, left, right)

struct SepMulti <: StringParser
    sep::String 
    content::Maybe{Vector{SuStringParser}}
end
SepMulti(sm::SepMulti, content::Maybe{Vector{SuStringParser}}) = SepMulti(sm.sep, content)

pass_bind(func::Function, x::Any, f::Maybe) = begin 
    f isa Nothing && return x 
    return func(x, f)
end

string_parse(str::String, parser::Clause) = begin 
    str = rm_strSpace(str)
    @assert str[1]==parser.left 
    @assert str[end]==parser.right
    return Clause(parser, pass_bind(string_parse, str[2:end-1], parser.content))
end

match_seq(str::String, sep::String) = str[1:min(length(sep), length(str))]==sep
find_sep_index(str::String, sep::String, index::Int)::Maybe{Int} = begin 
    length(str)==0 && return nothing
    match_seq(str, sep) && return index 
    return find_sep_index(str[2:end], sep, index+1)
end
find_unique_sep_index(str::String, sep::String)::Int = begin 
    maybe_index = find_sep_index(str, sep, 1)
    @assert bind(maybe_index, i->find_sep_index(str[i+1:end], sep, 1)) isa Nothing 
    return maybe_index
end

string_parse(str::String, parser::SepTwo) = begin 
    str = rm_strSpace(str)
    index = find_unique_sep_index(str, parser.sep)
    left = str[1:index-1] |> rm_strSpace
    right = str[index+length(parser.sep):end] |> rm_strSpace
    return SepTwo(parser, pass_bind(string_parse, left, parser.left), pass_bind(string_parse, right, parser.right))
end

"""
(⋄) is the Maybe{Vector} version of (++) operator
Note that we could not use (++) operator name here, since a::Vector ++ b::Vector is already defined, and actually the compiler can not distinguish between Vector and a Maybe{Vector}"""
(⋄)(s1::Maybe{Vector{String}}, s2::Maybe{Vector{String}}) = chain(s1, s2, (++), Val(:pass))
parse_sepmulti(str::String, sep::String)::Maybe = begin
    str = rm_strSpace(str)
    length(str)==0 && return nothing
    maybe_index = find_sep_index(str, sep, 1)
    return bind(maybe_index, i-> [str[1:i-1]] ⋄ parse_sepmulti(str[i+1:end], sep))
end

string_parse(str::String, parser::SepMulti) = begin 
    str = rm_strSpace(str)
    @assert str[end]!==parser.sep 
    str = str*parser.sep
    content = parse_sepmulti(str, parser.sep)
    map_string_parse(xs, ys) = map(1:length(xs)) do  i
        x, y = xs[i], ys[i]
        return string_parse(x, y)
    end
    return SepMulti(parser, Vector{SuStringParser}(chain(content, parser.content, map_string_parse, Val(:pass))))
end


analyze_parser(parser::StringParser) = begin 
    from_list = parser.content.left.content
    to = parser.content.right
    @assert from_list isa Vector
    @assert to isa String
    return from_list, to
end

# to_chars(str::String)::Vector{Char} = begin 
#     length(str)==0 && return 
#     x, xs = str[1], str[2:end]
#     return ['x']
# end
to_chars(str::String)::Vector{Char} = foldl((x,y)->x++[y], str; init=Char[])


"""for example, 
    eincode: 
        (ij, jk -> ik)
        (ij, jk, kl -> il)
        (ijk, ikl -> jl)
"""
eincode_parse(eincode::String) = begin 
    parser = Clause('(', ')', SepTwo("->", SepMulti(",", nothing), nothing))
    parser = string_parse(eincode, parser)
    from_list, to = analyze_parser(parser)
    @show from_list, to
    to_merge = setdiff(reduce((++), map(to_chars, from_list)) |> unique, to_chars(to))
    @show to_merge
end

einsum(eincode::String, nsa::NSTensor, nsb::NSTensor) = begin 

end


eincode_parse("(ijk, ikl -> jl)")

# Meta.parse("(ijk, ikl) -> jl")