using Hyperopt
using LinearAlgebra
using Statistics
using Distances
using Determinantal
using Clustering
using MultivariateStats
using LowRankApprox
using DataFrames
using Dagger
using IterTools
using BenchmarkTools
using Random
using Plots
using JLD
using Colors

Random.seed!(1234)

# Dataset: ACE descriptors #####################################################

ds = JLD.load("HfO2_figshare_form_random.jld")

# Energies
ged_tr = ds["ace_ged_tr"].parent'
es_tr  = ds["es_tr"]
ged_te = ds["ace_ged_te"].parent'
es_te  = ds["es_te"]

# Forces
#gfd_tr = ds["ace_gfd_tr"]
#fs_tr  = ds["fs_tr"]
#gfd_te = ds["ace_gfd_te"]
#fs_te  = ds["fs_te"]

# Linear system
#A = [ged_tr; gfd_tr]
#b = [es_tr; fs_tr]
#N, M = size(A)

#rows = randperm(N)[1:100]
#A = A[rows, :]
#b = b[rows]
#N, M = size(A)

#A_te = [ged_te; gfd_te]
#b_te = [es_te; fs_te]
#N_te, M_te = size(A_te)

#A = ged_tr.parent'
#b = es_tr
#N, M = size(A)

#A_te = ged_te.parent'
#b_te = es_te
#N_te, M_te = size(A)

A_tr = ged_tr
b_tr = es_tr
N_tr, M_tr = size(A_tr)

A_te = ged_te
b_te = es_te
N_te, M_te = size(A_te)


# Data reduction algorithms ####################################################

# Thin wrapper to DPP to reduce samples (rows)
function dpp_r(A, N′)
    # Compute a kernel matrix for the points in x
    #L = [ exp(-norm(a-b)^2) for a in eachcol(A'), b in eachcol(A') ]
    L = pairwise(Euclidean(), A')
    
    # Form an L-ensemble based on the L matrix
    dpp = EllEnsemble(L)
    
    # Scale so that the expected size is N′
    rescale!(dpp, N′)
    
    # A sample from the DPP (indices)
    r = Determinantal.sample(dpp)

    r′ = @views r
    if length(r) > N′
        r′ = @views r[1:N′]
    end

    return r′
end

# Thin wrapper to kmeans to reduce samples (rows)
function kmeans_r(A, N′)
    c = kmeans(A', 5; maxiter=200)
    a = c.assignments # get the assignments of points to clusters
    n_clusters = maximum(a)
    clusters = [findall(x->x==i, a) for i in 1:n_clusters]
    r = reduce(vcat, Clustering.sample.(clusters, [N′ ÷ length(clusters)]))
    return r
end

# Thin wrapper to dbscan to reduce samples (rows)
function dbscan_r(A, N′)
    # Create clusters using dbscan
    c = dbscan(A', 10; min_neighbors = 3, min_cluster_size = 20, metric=Euclidean())
    a = c.assignments # get the assignments of points to clusters
    n_clusters = maximum(a)
    clusters = [findall(x->x==i, a) for i in 1:n_clusters]
    r = reduce(vcat, Clustering.sample.(clusters, [N′ ÷ length(clusters)]))
    return r
end

# Thin wrapper to CUR to reduce samples (rows)
function cur_r(A, N′)
    r, _ = cur(A)
    r′ = @views r
    if length(r) > N′
        r′ = @views r[1:N′]
    end
    return r′
end

# Thin wrapper to CUR to reduce features (cols)
function cur_c(A, M′)
    _, c = cur(A)
    c′ = @views c[1:M′]
    return c′
end

# Thin wrapper to PCA to reduce features (cols)
function pca_c(A, M′)
    pca_model = fit(PCA, A, maxoutdim = M′)
    A′ = predict(pca_model, A')'
    return A′
end

# Error function ###############################################################

function error(A, b, A_te, b_te)
    X = A \ b
    mae = mean(abs.(A_te * X .- b_te))
    return mae
end

# Optimization #################################################################

#we, wt, ws = 100.0, 1.0, 1e-6
precs = [Float32, Float64]
samplers = [cur_r, dpp_r, kmeans_r, dbscan_r]
ho = @hyperopt for  i           = 40,
                    sampler     = RandomSampler(),
                    prec        = precs,
                    sample_red  = samplers,
                    N′          = Integer.(round.(LinRange(M_tr+1, N_tr-1, 6))) #N_tr÷10:10:N_tr
                    #feature_red = [], #[cur_c, pca_c],
                    #M′          = M_tr÷10:10:M_tr
    M′= M_tr
    print(  i, "\t",
            prec, "\t", 
            sample_red, "\t",
            N′, "\t", "\n"
            #feature_red, "\t", 
            #M′, "\n"
            )

    #rows, cols = 1:N, 1:M
    rows, cols,  = :, :;
    time = @elapsed begin
        # Precision
        A′, b′ = prec.(A_tr), prec.(b_tr)

        # Sample reduction
        rows = sample_red(A′, N′)

        # Feature reduction
        #cols = feature_red(A′, M′)

        # Reduction of samples and features
        A_tr′′, b_tr′′ = @views A_tr[rows, cols], b_tr[rows]
    end

    # Error
    A_te′′, b_te′′ = @views A_te[:, cols], b_te[:]
    err = error(A_tr′′, b_tr′′, A_te′′, b_te′′)

    # New dataset size
    #data_size = N′ * M′ * sizeof(prec)

    # Multi-objetive target
   # target = we * err + ws * data_size
#    if err < 0.01 
#       target = ws * data_size
#    else
#       target = we * err + ws * data_size
#    end

    # Multi-objetive target
    #target, data_size, err, time
    err, N′, time
end

# dataset
# sampling algorithm
# distance
# data type
# sample size
# ->
# error
# sampling time
# memory usage

# Print results ################################################################
ho

# Plot #########################################################################

errors = map(x->x[1], ho.results)
sample_sizes  = map(x->x[2], ho.results)
ss_times = map(x->x[3], ho.results)
scatter()
ms = [:circle, :rect, :star5, :star4, :diamond, :hexagon, :cross, :xcross, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, :star7, :star8, :vline, :hline, :+, :x]
for (i, (s, p)) in enumerate(product(samplers, precs))
    ind = findall(x->(s in x) && (p in x), ho.history)
    scatter!(sample_sizes[ind],
            errors[ind]; 
            zcolor=ss_times[ind], color=palette(:berlin, 15), label="$s, $p",
            xlabel = "Training sample size",
            ylabel = "MAE | eV",
            colorbar_title = "Time", 
            markershape=ms[i],dpi=300)
            # colorbar_title = "Data reduction time | s", dpi=300)
end
savefig("data-reduction-benchmark.png")

