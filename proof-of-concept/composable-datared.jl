### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ ff4783de-d009-11ee-029b-d772bc206b48
begin
using Statistics
using MultivariateStats
using LowRankApprox
using Dagger
using BenchmarkTools
using Random
end

# ╔═╡ 256e824a-4835-47ae-b92c-d67d8552b36a
md"""
# Composable data reduction algorithms
"""

# ╔═╡ 46740a29-9f96-4333-9082-bb83e3d5007f
md"""
In this example we will use a data set consisting of N samples and N target values, where each sample has M features, to train a linear model. We will then reduce the number of features and samples in the dataset using PCA and CUR, respectively, and retrain the linear model. Finally, we will measure the impact on accuracy and system size.
"""

# ╔═╡ 5221223d-c449-4fa5-931c-f6a1f04c327c
md"""
## Creation of synthetic data
"""

# ╔═╡ a94cce61-ebe8-4b93-bc32-c12134fe3163
Random.seed!(1234)

# ╔═╡ 63a58cfd-9124-4255-bfdd-9c83cc625ae0
N, M = 10000, 1000 # No. of samples(rows) and no. of features(cols).

# ╔═╡ d9bc98a8-ffc0-4777-b4cc-25cec3a1158f
m = rand(N, M) .- 0.5 # Dataset (lhs): stack of N samples and M features per sample

# ╔═╡ 1f132c45-e6a5-4e5e-b437-f30458b4b03b
b = rand(N) # Dataset (rhs): target or true values

# ╔═╡ 069e72d7-b05c-4a80-95f5-84c6df8d0118
md"""
## Single reduction algorithm: reduce dataset features with PCA
"""

# ╔═╡ 4ea07747-a829-4e96-96b2-9a5cdeb44d01
N′ = 100 # reduced no. of features(cols)

# ╔═╡ b56957ce-ce50-44f4-8e42-f595a1c9ec5a
m_pca_model = fit(PCA, m', maxoutdim = N′)

# ╔═╡ 5bf8cbeb-1993-4dd9-b92c-9d458846322a
m_pca = predict(m_pca_model, m')' # m_pca_model.proj' * (m' .- m_pca_model.mean)

# ╔═╡ efc40cf2-dfd7-489e-935f-a8d1531cf9f0
md"""
## Composable reduction algorithms: reduce observations with CUR and features with PCA
"""

# ╔═╡ e3a63cf0-fe7f-4945-893b-75695f9d203e
begin
# m1 = @view m[1:N÷2, :]
# m2 = @view m[N÷2+1:end, :]
# rows1, cols1 = cur(m1)
# rows2, cols2 = cur(m2)
# m_cur = vcat(m1[rows1[1:N′÷2], :],
# 	         m2[rows2[1:N′÷2], :])
end

# ╔═╡ e8851b24-c06f-4b89-8a8a-6b9d7eb57ef9
begin
rows, cols = cur(m)
m_cur = @view m[rows, :]
end

# ╔═╡ c4f1b94c-9b91-4657-a3a1-fc2774c4e39f
md"""
Apply PCA to merged reduced dataset.
"""

# ╔═╡ c7e71bc5-1c04-467e-92a6-c13348f51ac9
m_pca_cur_model = fit(PCA, m_cur', maxoutdim = N′)

# ╔═╡ 75d52ea0-99e8-48de-870d-a85b95966a1a
m_pca_cur = predict(m_pca_cur_model, m_cur')'

# ╔═╡ 3f2b3e22-e183-4c6b-82ab-9c5f11ec2171
md"""
## Accuracy
"""

# ╔═╡ edb4e686-d699-4666-ad01-4c815535deeb
md"""
### MSE of linear model training applied to whole dataset
"""

# ╔═╡ fc3fafdf-ecc3-4593-9111-cf3c5dc7d64b
X = m \ b

# ╔═╡ 841056ad-2513-462f-b9dd-36a0b34fd209
mse = mean((m * X .- b).^2)

# ╔═╡ 24d2694e-d24c-4ef9-ae57-0968201b1c0f
md"""
### MSE of linear model training using dataset with reduced features by PCA
"""

# ╔═╡ 9b3dcb9d-f2f0-4e59-a3b5-355968ea4aac
md"""
Error is preserved using data reduction.
"""

# ╔═╡ 39b4e89c-f3f2-43b9-a769-174788d3185c
X_pca = m_pca \ b 

# ╔═╡ 0d3f7aae-0d10-4e32-b7b4-b21c0b8c855b
mse_pca = mean((m_pca * X_pca .- b).^2)

# ╔═╡ cd102ada-75ea-42d9-b45c-412a88105d86
md"""
### MSE of linear model training using dataset with reduced samples by CUR and reduced features by PCA
"""

# ╔═╡ 7b014544-e52a-48ce-852a-55765c7d0d0e
md"""
Error is maintained or reduced using data reduction.
"""

# ╔═╡ 7ae1a94c-8249-4ca4-907d-5a1951363f6e
#b_cur = vcat(b[rows1[1:N′÷2]],
#	         b[rows2[1:N′÷2].+N′÷2])
b_cur = b[rows]

# ╔═╡ 07ef825e-9d96-4919-ac4f-44bcac463d5c
X_pca_cur = m_pca_cur \ b_cur

# ╔═╡ 245b5a19-e611-4868-88d6-1e8b6dfa5b13
mse_pca_cur = mean((m_pca_cur * X_pca_cur .- b_cur).^2)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Dagger = "d58978e5-989f-55fb-8d15-ea34adc7bf54"
LowRankApprox = "898213cb-b102-5a47-900c-97e73b919f73"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
BenchmarkTools = "~1.4.0"
Dagger = "~0.18.7"
LowRankApprox = "~0.5.5"
MultivariateStats = "~0.10.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.0"
manifest_format = "2.0"
project_hash = "24cd96a89ad290eb2894e1329443d308af7a1048"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1f03a9fa24271160ed7e73051fba3c1a759b53f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.4.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "d2c021fbdde94f6cdaa799639adfeeaa17fd67f5"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.13.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.Dagger]]
deps = ["DataStructures", "Distributed", "Graphs", "LinearAlgebra", "MacroTools", "MemPool", "PrecompileTools", "Profile", "Random", "Requires", "ScopedValues", "Serialization", "SharedArrays", "SparseArrays", "Statistics", "StatsBase", "TimespanLogging", "UUIDs"]
git-tree-sha1 = "d1dcc122ea55b68358c75eda86490e910c68d843"
uuid = "d58978e5-989f-55fb-8d15-ea34adc7bf54"
version = "0.18.7"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "ac67408d9ddf207de5cfa9a97e114352430f01ed"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.16"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "899050ace26649433ef1af25bc17a815b3db52b7"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.9.0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Inflate]]
git-tree-sha1 = "ea8031dea4aff6bd41f1df8f2fdfb25b33626381"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.4"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5fdf2fe6724d8caabf43b557b84ce53f3b7e2f6b"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.0.2+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LowRankApprox]]
deps = ["FFTW", "LinearAlgebra", "LowRankMatrices", "Nullables", "Random", "SparseArrays"]
git-tree-sha1 = "031af63ba945e23424815014ba0e59c28f5aed32"
uuid = "898213cb-b102-5a47-900c-97e73b919f73"
version = "0.5.5"

[[deps.LowRankMatrices]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "7c8664b2f3d5c3d9b77605c03d53b18813e79b0f"
uuid = "e65ccdef-c354-471a-8090-89bec1c20ec3"
version = "1.0.1"

    [deps.LowRankMatrices.extensions]
    LowRankMatricesFillArraysExt = "FillArrays"

    [deps.LowRankMatrices.weakdeps]
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "72dc3cf284559eb8f53aa593fe62cb33f83ed0c0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.0.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.MemPool]]
deps = ["DataStructures", "Distributed", "Mmap", "Random", "ScopedValues", "Serialization", "Sockets"]
git-tree-sha1 = "60dd4ac427d39e0b3f15b193845324523ee71c03"
uuid = "f9f48841-c794-520a-933b-121f7ba6ed94"
version = "0.4.6"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Nullables]]
git-tree-sha1 = "8f87854cc8f3685a60689d8edecaa29d2251979b"
uuid = "4d1e1d77-625e-5b40-9113-a560ec7a8ecd"
version = "1.0.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c27d546a4749c81f70d1fabd604da6aa5054e3d2"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.2.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "7b0e9c14c624e435076d19aea1e5cbdec2b9ca37"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.2"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TimespanLogging]]
deps = ["Distributed", "Profile"]
git-tree-sha1 = "51be7dd35b0c8a5a613dc7af272d587ea6943d24"
uuid = "a526e669-04d3-4846-9525-c66122c55f63"
version = "0.1.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.7.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═ff4783de-d009-11ee-029b-d772bc206b48
# ╟─256e824a-4835-47ae-b92c-d67d8552b36a
# ╟─46740a29-9f96-4333-9082-bb83e3d5007f
# ╟─5221223d-c449-4fa5-931c-f6a1f04c327c
# ╠═a94cce61-ebe8-4b93-bc32-c12134fe3163
# ╠═63a58cfd-9124-4255-bfdd-9c83cc625ae0
# ╠═d9bc98a8-ffc0-4777-b4cc-25cec3a1158f
# ╠═1f132c45-e6a5-4e5e-b437-f30458b4b03b
# ╟─069e72d7-b05c-4a80-95f5-84c6df8d0118
# ╠═4ea07747-a829-4e96-96b2-9a5cdeb44d01
# ╠═b56957ce-ce50-44f4-8e42-f595a1c9ec5a
# ╠═5bf8cbeb-1993-4dd9-b92c-9d458846322a
# ╟─efc40cf2-dfd7-489e-935f-a8d1531cf9f0
# ╠═e3a63cf0-fe7f-4945-893b-75695f9d203e
# ╠═e8851b24-c06f-4b89-8a8a-6b9d7eb57ef9
# ╟─c4f1b94c-9b91-4657-a3a1-fc2774c4e39f
# ╠═c7e71bc5-1c04-467e-92a6-c13348f51ac9
# ╠═75d52ea0-99e8-48de-870d-a85b95966a1a
# ╟─3f2b3e22-e183-4c6b-82ab-9c5f11ec2171
# ╟─edb4e686-d699-4666-ad01-4c815535deeb
# ╠═fc3fafdf-ecc3-4593-9111-cf3c5dc7d64b
# ╠═841056ad-2513-462f-b9dd-36a0b34fd209
# ╟─24d2694e-d24c-4ef9-ae57-0968201b1c0f
# ╟─9b3dcb9d-f2f0-4e59-a3b5-355968ea4aac
# ╠═39b4e89c-f3f2-43b9-a769-174788d3185c
# ╠═0d3f7aae-0d10-4e32-b7b4-b21c0b8c855b
# ╟─cd102ada-75ea-42d9-b45c-412a88105d86
# ╟─7b014544-e52a-48ce-852a-55765c7d0d0e
# ╠═7ae1a94c-8249-4ca4-907d-5a1951363f6e
# ╠═07ef825e-9d96-4919-ac4f-44bcac463d5c
# ╠═245b5a19-e611-4868-88d6-1e8b6dfa5b13
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
