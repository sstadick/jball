module JBall

using DataFrames
using Statistics
using LinearAlgebra
using MultivariateStats
using NFLData
using Plots
using Clustering

wr_cols = [
    # "carries",
    "receptions",
    # "targets",
    "receiving_yards",
    # "receiving_tds",
    # "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_first_downs",
    "receiving_epa",
    "racr", # Ratio of receiving yards by total air yards
    "target_share",
    # "air_yards_share",
    "wopr", # Weighted combination of the share of team targets a player receives and the share of team air yards
    "fantasy_points",
    "fantasy_points_ppr"
]

rb_cols = [
    "carries",
    "rushing_yards",
    # "rushing_tds",
    "rushing_first_downs",
    "rushing_epa",
    # "rushing_2pt_conversions",
    "receptions",
    "targets",
    "receiving_yards",
    "fantasy_points",
    "fantasy_points_ppr"
]

qb_cols = [
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "sacks",
    "sack_yards",
    "sack_fumbles",
    "sack_fumbles_lost",
    "passing_air_yards",
    "passing_yards_after_catch",
    "passing_first_downs",
    "passing_epa",
    "passing_2pt_conversions",
    "pacr",
    "carries",
    "rushing_yards",
    "rushing_first_downs",
    "rushing_epa",
    "fantasy_points",
    "fantasy_points_ppr"
]

"""
    assign_clusters(X::AbstractMatrix{<:Real}, R::KmeansResult; kwargs...) -> Vector{Int}
Assign the samples specified as the columns of `X` to the corresponding clusters from `R`.
# Arguments

- `X`: Input data to be clustered.
- `R`: Fitted clustering result.
# Keyword arguments
- `distance`: SemiMertric used to compute distances between vectors and clusters centroids.
- `pairwise_computation`: Boolean specifying whether to compute and store pairwise distances.
"""
function assign_clusters(
    X::AbstractMatrix{T}, 
    R::KmeansResult;
    distance::Clustering.SemiMetric = Clustering.SqEuclidean(),
    pairwise_computation::Bool = true) where {T} 

    if pairwise_computation 
        Xdist = Clustering.pairwise(distance, X, R.centers, dims=2)
        cluster_assignments = partialsortperm.(eachrow(Xdist), 1)
    else
        cluster_assignments = zeros(Int, size(X, 2))        
        Threads.@threads for n in axes(X, 2)
            min_dist = typemax(T)
            cluster_assignment = 0

            for k in axes(R.centers, 2)
                dist = distance(@view(X[:, n]), @view(R.centers[:, k]))
                if dist < min_dist
                    min_dist = dist
                    cluster_assignment = k
                end
            end
            cluster_assignments[n] = cluster_assignment
        end
    end

    return cluster_assignments
end

"""
    group_and_combine_stats(stats; start_week=1, end_week=18)

Group the stats by `player_name` and `player_id`, then merge all the stats between `completions` and `special_teams_tds` by converting them to averages.
"""
function group_and_combine_stats(stats, season, position, cols; start_week=1, end_week=18)
    ret = combine(
                groupby(
                    filter([:season, :position, :week] => (s, p, w) -> s == season && p == position && (w >= start_week && w <= end_week), stats),
                    [:player_name, :player_id]
                ),
            cols .=> mean
        )
end

function do_pca_and_plot(last_year, this_year, pos)
    select!(last_year, Not([:fantasy_points_mean, :fantasy_points_ppr_mean]))
    select!(this_year, Not([:fantasy_points_mean, :fantasy_points_ppr_mean]))
    model = fit(PCA, transpose(to_matrix(normcol!(last_year))); maxoutdim=5)

    this_year = fix_missing_strings!(this_year)
    for r in eachrow(this_year)
        r.player_name = "$(r.player_name)'24"
    end

    last_year_transformed = MultivariateStats.transform(model, transpose(to_matrix(normcol!(last_year))))
    this_year_transformed = MultivariateStats.transform(model, transpose(to_matrix(normcol!(this_year))))

    # p = plot( model.prinvars, linewidth=4, title="eigenvalues", size=(600,200), legend=false)
    # p = scatter(last_year_transformed[1,:], last_year_transformed[2,:], color=:blue, markersize=1, markerstrokewidth=0, series_annotations=text.(last_year.player_name, 4, :blue, :bottom), label="2023")
    # scatter!(this_year_transformed[1,:], this_year_transformed[2,:], color=:orange, markersize=1, markerstrokewidth=0, series_annotations=text.(this_year.player_name, 4, :orange, :bottom), label="2024")

    result = kmeans(last_year_transformed, 10; maxiter=200, display=:iter)
    predicted = assign_clusters(this_year_transformed, result)
    p = scatter(
        last_year_transformed[1,:],
        last_year_transformed[2,:],
        color=:lightrainbow,
        marker_z=result.assignments,
        markersize=2,
        markerstrokewidth=0,
        series_annotations=text.(last_year.player_name, 2, :blue, :bottom),
        label="2023",
        legend=false,
        dpi=1000
    )
    scatter!(
        this_year_transformed[1,:],
        this_year_transformed[2,:],
        color=:lightrainbow,
        marker_z=predicted,
        markersize=2,
        markerstrokewidth=0,
        series_annotations=text.(this_year.player_name, 2, :orange, :bottom),
        label="2024",
        legend=false
    )
    display(p)
    savefig("/tmp/$(pos).png")

    this_year.cluster = predicted
    last_year.cluster = result.assignments

    return (this_year, last_year)

    # TODO: create groupings of players so that projected points can be created for each cluster based on last year, the output this years players in the their clusters next to last years players


end

function run(pos=:wr)
    stats = load_player_stats()
    current_week = 4

    wrs_2023 = group_and_combine_stats(stats, 2023, "WR", wr_cols, start_week=1, end_week=18)
    rbs_2023 = group_and_combine_stats(stats, 2023, "RB", rb_cols, start_week=1, end_week=18)
    tes_2023 = group_and_combine_stats(stats, 2023, "TE", wr_cols, start_week=1, end_week=18)
    qbs_2023 = group_and_combine_stats(stats, 2023, "QB", qb_cols, start_week=1, end_week=18)


    # https://juliastats.org/MultivariateStats.jl/dev/pca/
    # https://medium.com/@avmantzaris/intro-to-principal-component-analysis-pca-for-email-spam-detection-using-julia-lang-73915036d7b3
    wrs_2024 = group_and_combine_stats(stats, 2024, "WR", wr_cols, start_week=max(current_week-3, 1), end_week=current_week)
    rbs_2024 = group_and_combine_stats(stats, 2024, "RB", rb_cols, start_week=max(current_week-3, 1), end_week=current_week)
    tes_2024 = group_and_combine_stats(stats, 2024, "TE", wr_cols, start_week=max(current_week-3, 1), end_week=current_week)
    qbs_2024 = group_and_combine_stats(stats, 2024, "QB", qb_cols, start_week=max(current_week-3, 1), end_week=current_week)

    # Analyze
    if pos == :rb
        (this_year_rbs, last_year_rbs) = do_pca_and_plot(deepcopy(rbs_2023), deepcopy(rbs_2024), :rb)
        rbs_2024_clustered = join_and_assign_clusters(rbs_2023, this_year_rbs)
        rbs_2023_clustered = join_and_assign_clusters(rbs_2023, last_year_rbs)
        group_clusters(rbs_2023_clustered, rbs_2024_clustered)
    elseif pos == :wr
        (this_year_wrs, last_year_wrs) = do_pca_and_plot(deepcopy(wrs_2023), deepcopy(wrs_2024), :wr)
        wrs_2024_clustered = join_and_assign_clusters(wrs_2023, this_year_wrs)
        wrs_2023_clustered = join_and_assign_clusters(wrs_2023, last_year_wrs)
        group_clusters(wrs_2023_clustered, wrs_2024_clustered)
    elseif pos == :te
        (this_year_tes, last_year_tes) = do_pca_and_plot(deepcopy(tes_2023), deepcopy(tes_2024), :te)
        tes_2024_clustered = join_and_assign_clusters(tes_2023, this_year_tes)
        tes_2023_clustered = join_and_assign_clusters(tes_2023, last_year_tes)
        group_clusters(tes_2023_clustered, tes_2024_clustered)
    elseif pos == :qb
        (this_year_qbs, last_year_qbs) = do_pca_and_plot(deepcopy(qbs_2023), deepcopy(qbs_2024), :qb)
        qbs_2024_clustered = join_and_assign_clusters(qbs_2023, this_year_qbs)
        qbs_2023_clustered = join_and_assign_clusters(qbs_2023, last_year_qbs)
        group_clusters(qbs_2023_clustered, qbs_2024_clustered)
    else
        print("Unknown position $pos")
    end
    # Get the full rull from stats for each and append the cluster to it

    # do_pca_and_plot(deepcopy(wrs_2023), deepcopy(wrs_2024))
    # do_pca_and_plot(deepcopy(tes_2023), deepcopy(tes_2024))
    # do_pca_and_plot(deepcopy(qbs_2023), deepcopy(qbs_2024))

    # TODO: kmeans
end

function group_clusters(last_year, this_year)
    last_year_points = combine(
                groupby(
                    this_year,
                    [:cluster]
                ),
            [:fantasy_points_mean, :fantasy_points_ppr_mean] .=> mean
        )
    
    print(last_year_points)

end

function join_and_assign_clusters(data, assignments)
    return innerjoin(data, assignments[:, [:player_id, :cluster]], on = :player_id)
end

function normcol!(x)
    for col in eachcol(x)
        if eltype(col) != Union{Missing,String} && eltype(col) != String
            fixed = coalesce.(col, 0.0)
            col ./= norm(fixed)
        end
    end
    return x
end

function fix_missing_strings!(x)
    for col in eachcol(x)
        if eltype(col) == Union{Missing,String}
            fixed = coalesce.(col, "")
            col = fixed
        end
    end
    return x
end


function to_matrix(df)
    mat = Matrix(df[:, 3:end])
    ret = replace(mat, NaN=>0.0)
    ret = replace(ret, missing=>0.0)
    return ret 
end

export group_and_combine_stats
export normcol! 
export to_matrix



end # module JBall


