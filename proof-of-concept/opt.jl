using Hyperopt
using Statistics
using MultivariateStats
using LowRankApprox
using Dagger
using BenchmarkTools
using Random
using Plots

Random.seed!(1234)

# Synthetic data ##############################################################

N, M = 1000, 100     # No. of samples(rows) and no. of features(cols).
ds = rand(N, M) .- 0.5 # Dataset (lhs): stack of N samples and M features per sample
b = rand(N)            # Dataset (rhs): target or true values

# Data reduction algorithms ###################################################

# Thin wrapper to CUR to reduce samples (rows)
function cur_r(A, b, N′)
    r, _ = cur(A)
    r′ = @views r
    if length(r) > N′
        r′ = @views r[1:N′]
    end
    return @views A[r′, :], b[r′]
end

# Thin wrapper to CUR to reduce features (cols)
function cur_c(A, M′)
    _, c = cur(A)
    c′ = @views c[1:M′]
    return @views A[:, c′]
end

# Thin wrapper to PCA to reduce features (cols)
function pca_c(A, N′)
    pca_model = fit(PCA, A, maxoutdim = N′)
    A′ = predict(pca_model, A')'
    return A′
end

# Error function ###############################################################

function error(A, b) # Here, I should use test data
    X = A \ b
    mean((A * X .- b).^2)
end

# Optimization ################################################################

we, wt, ws = 1.0, 1.0, 1e-5
ho = @hyperopt for  i           = 20,
                    sampler     = RandomSampler(),
                    prec        = [Float32, Float64],
                    sample_red  = [cur_r],
                    N′          = N÷10:10:N,
                    feature_red = [cur_c, pca_c],
                    M′          = M÷10:10:M
   
    print(  i, "\t",
            prec, "\t", 
            sample_red, "\t",
            N′, "\t",
            feature_red, "\t", 
            M′, "\n")

    time = @elapsed begin
        # Precision
        ds′, b′ = prec.(ds), prec.(b)

        # Sample reduction
        ds′′, b′ = sample_red(ds′, b′, N′)

        # Feature reduction
        ds′′′ = feature_red(ds′′, M′)
    end

    # Error
    err = error(ds′′′, b′)

    # New dataset size
    size = N′ * M′ * sizeof(prec)

    # Multi-objetive target
    target = we * err + ws * size

    # Multi-objetive target
    target, size, err, time
end

# Print results ###############################################################
ho

# Plot ########################################################################

sizes  = map(x->x[2], ho.results) / 1024^2
errors = map(x->x[3], ho.results)
times  = map(x->x[4], ho.results)
scatter(sizes, errors; zcolor=times, label="",
        xlabel = "Dataset size | MB", ylabel = "MSE",
        colorbar_title = "Data reduction time | s")
savefig("data-reduction-benchmark.png")