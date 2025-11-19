/*
    Multiplicative model
    Copyright (C) 2023 Lucas Czech and Xing Wu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Contact:
    Lucas Czech <lczech@carnegiescience.edu>
    Xing Wu <wxu@carnegiescience.edu>
    Department of Plant Biology, Carnegie Institution For Science
    260 Panama Street, Stanford, CA 94305, USA
*/

// #include "genesis/genesis.hpp"
#include "genesis/utils/containers/matrix.hpp"
#include "genesis/utils/containers/matrix/operators.hpp"
#include "genesis/utils/containers/matrix/simple_reader.hpp"
#include "genesis/utils/io/output_target.hpp"
#include "genesis/utils/math/moments.hpp"
#include "genesis/utils/math/ranking.hpp"
#include "genesis/utils/text/string.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace genesis;
using namespace genesis::utils;

// =================================================================================================
//     Environment
// =================================================================================================

struct Data
{
    // Data
    Matrix<unsigned char> x; // num_indiv x num_snps
    std::vector<double> y;   // num_indiv
    Matrix<double> c;        // num_indiv x num_covar

    // Helpers for intuitive naming
    size_t num_indiv;
    size_t num_snps;
    size_t num_covar;
};

struct Runparams
{
    size_t burnin_iterations = 2000;
    size_t regular_iterations = 10000;
    size_t sanity_iterations = 100;
    size_t test_convergence_start = 10000;
    size_t test_convergence_interval = 1000;
    size_t test_convergence_stop = 100000;
};

struct Hyperparams
{
    // Distribution hyperparameters.
    // pi_b is initialized to ratio num_snps / pi_b_ratio
    double pi_a = 1.0;
    //double pi_b = 1.0;
    double pi_b_ratio = 10.0;
    double sigma_1_a = 1.0;
    double sigma_1_b = 1.0;
    double sigma_e_a = 1.0;
    double sigma_e_b = 1.0;

    // Sample validity
    double min_beta = 0.05;
    double large_beta = 0.3;

    // Convergence criteria
    size_t num_top_betas = 5;
    double geweke_first = 0.1;
    double geweke_last = 0.5;
    size_t geweke_intervals = 20;
    double convergence_max_zscore = 1.5;
};

struct Posteriors
{
    std::vector<double> alpha;
    std::vector<double> beta;
    std::vector<int> gamma;
    double pi = 0.0;
    double sigma_1 = 0.0;
    double sigma_e = 0.0;
};

struct State
{
    // Random state
    // TODO: other engines are faster - if they are good enough for MCMC, we might want to switch
    std::random_device rand_dev;
    std::default_random_engine rand_gen{ rand_dev() };

    // Intermediate values kept between the sampling update functions to avoid recomputation
    std::vector<double> c_col_sum_squared;
    std::vector<double> circle_product_x_beta;
    std::vector<double> product_c_alpha;

    // Intermediate values per snp
    std::vector<double> snp_norm;
    std::vector<double> snp_mean;
    std::vector<double> snp_variance;
};

struct Statistics
{
    size_t polygenicity            = 0;
    double genetic_var             = 0.0;
    double pheno_var               = 0.0;
    double large_beta_ratio        = 0.0;
    double large_beta_heritability = 0.0;
    double total_heritability      = 0.0;
};

struct TraceEntry
{
    size_t iteration = 0;
    Posteriors post;
    Statistics stats;
    std::vector<double> top_betas;
};

struct Trace
{
    std::vector<TraceEntry> entries;
};

// =================================================================================================
//     Input Data
// =================================================================================================

// -------------------------------------------------------------------------
//     check_matrix_range
// -------------------------------------------------------------------------

template<typename T>
void check_matrix_range( Matrix<T> const& matrix, T min = 0, T max = 0 )
{
    for( auto e : matrix ) {
        if( ! std::isfinite( e )) {
            throw std::runtime_error( "Matrix has non-finite values" );
        }
        if(( min != 0 || max != 0 ) && ( e < min || e > max )) {
            throw std::runtime_error(
                "Matrix has values outside of [" + std::to_string( min ) + ", " +
                std::to_string( max ) + "]"
            );
        }
    }
}

// -------------------------------------------------------------------------
//     read_input_data
// -------------------------------------------------------------------------

Data read_input_data(
    std::string const& x_file,
    std::string const& y_file,
    std::string const& c_file
) {
    Data data;

    // Set up fast readers for simple matrices
    auto unsigned_char_reader = MatrixSimpleReader<unsigned char>();
    unsigned_char_reader.parse_value_functor( parse_unsigned_integer<unsigned char> );
    auto double_reader = MatrixSimpleReader<double>();
    double_reader.parse_value_functor( parse_float<double> );

    // Read x (genotype matrix)
    LOG_INFO << "reading X";
    data.x = unsigned_char_reader.read( from_file( x_file ));
    LOG_INFO << "x[" << data.x.rows() << "," << data.x.cols() << "]";
    LOG_INFO << print( data.x );
    check_matrix_range<unsigned char>( data.x, 0, 2 );

    // Read y (phenotype vector)
    LOG_INFO << "reading Y";
    auto y_mat = double_reader.read( from_file( y_file ));
    if( y_mat.cols() != 1 ) {
        throw std::invalid_argument( "Input y has more than one column" );
    }
    LOG_INFO << "y[" << y_mat.rows() << "]";
    LOG_INFO << print( y_mat );
    check_matrix_range( y_mat );
    data.y = std::move( y_mat.data() );

    // Read c (covariates)
    LOG_INFO << "reading C";
    data.c = double_reader.read( from_file( c_file ));
    LOG_INFO << "c[" << data.c.rows() << "," << data.c.cols() << "]";
    LOG_INFO << print( data.c );
    check_matrix_range( data.c );

    // Safety checks
    if( data.x.rows() != data.y.size() || data.x.rows() != data.c.rows() ) {
        throw std::invalid_argument( "Input data dimensions do not match" );
    }
    if( data.x.rows() == 0 || data.x.cols() == 0 || data.c.cols() == 0 ) {
        throw std::invalid_argument( "Input data is empty" );
    }

    // Set some nicely named helpers
    data.num_indiv = data.x.rows();
    data.num_snps  = data.x.cols();
    data.num_covar = data.c.cols();

    return data;
}

// =================================================================================================
//     Helper Functions
// =================================================================================================

// -------------------------------------------------------------------------
//     draw_pi
// -------------------------------------------------------------------------

template< class Generator >
double draw_pi( double a, double b, Generator& gen )
{
    // Need to get a beta distrib by using two gamma distribs,
    // see https://stackoverflow.com/a/10359049
    std::gamma_distribution<> dist_gamma_a( a, 1.0 );
    std::gamma_distribution<> dist_gamma_b( b, 1.0 );
    auto const draw_a = dist_gamma_a( gen );
    auto const draw_b = dist_gamma_b( gen );
    return draw_a / ( draw_a + draw_b );
}

// -------------------------------------------------------------------------
//     draw_sigma_std_dev
// -------------------------------------------------------------------------

template< class Generator >
double draw_sigma_std_dev( double a, double b, Generator& gen )
{
    // see https://stackoverflow.com/a/10359049
    std::gamma_distribution<> dist_gamma( a, 1.0 / b );
    auto const sigma_1_neg2 = dist_gamma( gen );
	return std::sqrt( 1.0 / sigma_1_neg2 );
}

// -------------------------------------------------------------------------
//     circle_product_matrix
// -------------------------------------------------------------------------

std::vector<double> circle_product_matrix(
    Matrix<unsigned char> const& x, std::vector<double> const& beta
) {
    // Make sure that everything has the right dimensions.
    assert( x.cols() == beta.size() );
    auto result = std::vector<double>( x.rows() );

    // Element-wise multiplication of each row of X with beta,
    // then multiply along the columns to get the product for each row.
    for( size_t r = 0; r < x.rows(); ++r ) {
        result[r] = 1.0;
        for( size_t c = 0; c < x.cols(); ++c ) {
            result[r] *= 1.0 + x( r, c ) * beta[c];
        }
    }
    return result;
}

// -------------------------------------------------------------------------
//     update_state_snp_cache
// -------------------------------------------------------------------------

void update_state_snp_cache( Data const& data, Posteriors const& post, State& state, size_t s )
{
    assert( data.num_indiv == data.x.rows() );
    assert( data.num_indiv == data.y.size() );
    assert( data.num_indiv == state.circle_product_x_beta.size() );
    assert( data.num_indiv == state.product_c_alpha.size() );
    assert( data.num_snps  == state.snp_norm.size() );
    assert( data.num_snps  == state.snp_variance.size() );
    assert( data.num_snps  == state.snp_mean.size() );
    assert( s < data.num_snps );

    // We need to accumulate over all individuals
    double snp_norm_x_beta_x = 0.0;
    double dot_prod_residuals = 0.0;
    for( size_t i = 0; i < data.num_indiv; ++i ) {
        // Compute helper variables.
        // circle_product_x_beta is the equivalent of circle_product_x_beta_negi in the Python code,
        // i.e., all _but_ the current SNP.
        auto const x_beta_x = state.circle_product_x_beta[i] * data.x(i, s);
        auto const residual = data.y[i] - state.product_c_alpha[i] - state.circle_product_x_beta[i];

        // Update our sums
        snp_norm_x_beta_x += x_beta_x * x_beta_x;
        dot_prod_residuals += residual * x_beta_x;
    }

    // Some constants
    auto const sigma_e_neg2 = std::pow( post.sigma_e, -2 );
    auto const sigma_1_neg2 = std::pow( post.sigma_1, -2 );

    // Compute the new mean and variance and norm of x beta x, and store them
    state.snp_norm[s]     = snp_norm_x_beta_x;
    state.snp_variance[s] = 1.0 / ( snp_norm_x_beta_x * sigma_e_neg2 + sigma_1_neg2 );
    state.snp_mean[s]     = state.snp_variance[s] * dot_prod_residuals * sigma_e_neg2;

    LOG_DBG << s << ": snp_norm_x_beta_x=" << snp_norm_x_beta_x << " " << "dot_prod_residuals=" << dot_prod_residuals << " " << "state.snp_norm[s]=" << state.snp_norm[s] << " " << "state.snp_variance[s]=" << state.snp_variance[s] << " " << "state.snp_mean[s]=" <<  state.snp_mean[s];
}

// -------------------------------------------------------------------------
//     refresh_state_cache_alpha
// -------------------------------------------------------------------------

void refresh_state_cache_alpha( Data const& data, Posteriors const& post, State& state )
{
    assert( data.num_indiv == state.product_c_alpha.size() );
    assert( data.num_covar == post.alpha.size() );

    // Init the covarite based values. Will be updated in each iteration.
    for( size_t i = 0; i < data.num_indiv; ++i ) {
        for( size_t c = 0; c < data.num_covar; ++c ) {
            state.product_c_alpha[i] += data.c(i, c) * post.alpha[ c ];
        }
    }
}

// -------------------------------------------------------------------------
//     refresh_state_cache_snp_values
// -------------------------------------------------------------------------

void refresh_state_cache_snp_values( Data const& data, Posteriors const& post, State& state )
{
    assert( data.num_snps == state.snp_norm.size() );
    assert( data.num_snps == state.snp_mean.size() );
    assert( data.num_snps == state.snp_variance.size() );

    // We need to do a bit of trickery here to compute the SNP norms based on all _but_ the current
    // SNP. Hence, remove its impact first, then do the norm, then move it back in.
    for( size_t s = 0; s < data.num_snps; ++s ) {
        // Remove the circ prod of the non-updated beta from current snp in our cache
        for( size_t i = 0; i < data.num_indiv; ++i ) {
            auto const circ_prod = data.x(i, s) * post.beta[s] + 1.0;
            state.circle_product_x_beta[i] /= circ_prod;
        }

        // Now we can update the cache values, using all _but_ the value we just removed
        update_state_snp_cache( data, post, state, s );

        // Add the updated value to current snp in our cache
        for( size_t i = 0; i < data.num_indiv; ++i ) {
            auto const circ_prod = data.x(i, s) * post.beta[s] + 1.0;
            state.circle_product_x_beta[i] *= circ_prod;
        }
    }
}

// -------------------------------------------------------------------------
//     refresh_state_cache
// -------------------------------------------------------------------------

void refresh_state_cache( Data const& data, Posteriors const& post, State& state )
{
    // Also init the other caches by calling their refresh functions
    state.circle_product_x_beta = circle_product_matrix( data.x, post.beta );
    refresh_state_cache_alpha( data, post, state );
    refresh_state_cache_snp_values( data, post, state );
}

// -------------------------------------------------------------------------
//     init_state_cache
// -------------------------------------------------------------------------

void init_state_cache( Data const& data, Posteriors const& post, State& state )
{
    // Init the caches.
    state.c_col_sum_squared.resize( data.num_covar, 0.0 );
    state.product_c_alpha.resize( data.num_indiv, 0.0 );
    state.snp_norm.resize( data.num_snps );
    state.snp_mean.resize( data.num_snps );
    state.snp_variance.resize( data.num_snps );

    // Pre-compute the per-column sum of squares of c.
    // This is constant throughout the run.
    for( size_t c = 0; c < data.num_covar; ++c ) {
        for( size_t i = 0; i < data.num_indiv; ++i ) {
            state.c_col_sum_squared[c] += data.c(i, c) * data.c(i, c);
        }
    }

    refresh_state_cache( data, post, state );
}

// =================================================================================================
//     Sampling
// =================================================================================================

// -------------------------------------------------------------------------
//     sample_pi
// -------------------------------------------------------------------------

double sample_pi( Hyperparams const& hyper, Posteriors const& post, State& state )
{
    // Sum the gammas, and compute new a and b
    size_t sum_gamma_p = 0;
    size_t sum_gamma_n = 0;
    for( auto g : post.gamma ) {
        assert( g == 0 || g == 1 );
        sum_gamma_p += g;
        sum_gamma_n += 1 - g;
    }
    auto const a_new = static_cast<double>( sum_gamma_p ) + hyper.pi_a;
    auto const b_new = static_cast<double>( sum_gamma_n ) + hyper.pi_b;

    // Draw from distribution
    return draw_pi( a_new, b_new, state.rand_gen );
}

// -------------------------------------------------------------------------
//     sample_sigma_1
// -------------------------------------------------------------------------

double sample_sigma_1( Hyperparams const& hyper, Posteriors const& post, State& state )
{
    // Compute new a and b
    auto const num_snps = post.gamma.size();
    size_t sum_gamma = 0;
    double sum_beta_gamma_sqr = 0.0;
    assert( num_snps == post.beta.size() );
    for( size_t s = 0; s < num_snps; ++s ) {
        sum_gamma += post.gamma[s];
        sum_beta_gamma_sqr += post.beta[s] * post.beta[s] * post.gamma[s];
    }
    auto const a_new = 0.5 * static_cast<double>( sum_gamma ) + hyper.sigma_1_a;
	auto const b_new = 0.5 * sum_beta_gamma_sqr + hyper.sigma_1_b;

    // Draw from distribution
    return draw_sigma_std_dev( a_new, b_new, state.rand_gen );
}

// -------------------------------------------------------------------------
//     sample_sigma_e
// -------------------------------------------------------------------------

double sample_sigma_e( Data const& data, Hyperparams const& hyper, State& state )
{
    assert( data.num_indiv == state.circle_product_x_beta.size() );
    assert( data.num_indiv == state.product_c_alpha.size() );

    // Compute sum of squared residuals
    double sum_square_resid = 0.0;
    for( size_t i = 0; i < data.num_indiv; ++i ) {
        auto const resid = data.y[i] - state.circle_product_x_beta[i] - state.product_c_alpha[i];
        sum_square_resid += resid * resid;
    }

    // Compute new a and b
    auto const a_new = static_cast<double>( data.num_indiv ) / 2.0 + hyper.sigma_e_a;
    auto const b_new = sum_square_resid / 2.0 + hyper.sigma_e_b;

    // Draw from distribution
    return draw_sigma_std_dev( a_new, b_new, state.rand_gen );
}

// -------------------------------------------------------------------------
//     sample_and_update_alpha
// -------------------------------------------------------------------------

void sample_and_update_alpha( Data const& data, Posteriors& post, State& state )
{
    assert( data.num_indiv == data.y.size() );
    assert( data.num_indiv == state.circle_product_x_beta.size() );
    assert( data.num_indiv == state.product_c_alpha.size() );
    assert( data.num_covar == post.alpha.size() );
    assert( data.num_covar == state.c_col_sum_squared.size() );

    for( size_t c = 0; c < data.num_covar; ++c ) {
        // Compute the dot product for all individuals.
        // If we have multiple covariates, we here remove the value of the current column
        // from the product_c_alpha cache. If we only have one covariate, this is skipped.
        double dot_prod = 0.0;
        if( data.num_covar == 1 ) {
            assert( c == 0 );
            for( size_t i = 0; i < data.num_indiv; ++i ) {
                auto const y_x_beta = data.y[i] - state.circle_product_x_beta[i];
                dot_prod += y_x_beta * data.c(i, c);
            }
        } else {
            for( size_t i = 0; i < data.num_indiv; ++i ) {
                state.product_c_alpha[i] -= data.c(i, c) * post.alpha[c];
                auto const c_alpha_negi = state.product_c_alpha[i];
                auto const y_negi = data.y[i] - c_alpha_negi - state.circle_product_x_beta[i];
                dot_prod += y_negi * data.c(i, c);
            }
        }

        // Compute new mean and var
        auto const sigma_e_neg2 = std::pow( post.sigma_e, -2 );
        auto const new_variance = 1.0 / ( state.c_col_sum_squared[c] * sigma_e_neg2 );
        auto const new_mean = new_variance * dot_prod * sigma_e_neg2;

        // Draw a new alpha
        std::normal_distribution<> dist_normal( new_mean, std::sqrt( new_variance ));
        post.alpha[c] = dist_normal( state.rand_gen );

        // We need to update the cached product across all rows,
        // either by recomputing (for one covar), or by adding back in the new alpha value
        // (for multiple covariates).
        if( data.num_covar == 1 ) {
            assert( c == 0 );
            for( size_t i = 0; i < data.num_indiv; ++i ) {
                state.product_c_alpha[i] = data.c(i, c) * post.alpha[c];
            }
        } else {
            for( size_t i = 0; i < data.num_indiv; ++i ) {
                state.product_c_alpha[i] += data.c(i, c) * post.alpha[c];
            }
        }
    }
}

// -------------------------------------------------------------------------
//     sample_and_update_gamma
// -------------------------------------------------------------------------

void sample_and_update_gamma( Data const& data, Posteriors& post, State& state )
{
    assert( data.num_snps == post.gamma.size() );

    // Some constants
    auto const sigma_e_neg2 = std::pow( post.sigma_e, -2 );
    auto const sigma_1_neg2 = std::pow( post.sigma_1, -2 );

    // Update gamma for all snps
    for( size_t s = 0; s < data.num_snps; ++s ) {

        // Compute the intermediate values
        auto const d = state.snp_norm[s] * sigma_e_neg2 / sigma_1_neg2 + 1.0;
        auto const f = std::sqrt( 1.0 / d );
        auto const v = state.snp_mean[s] * state.snp_mean[s];
        auto const a = f * std::exp( 0.5 * v / state.snp_variance[s] );
        auto const b = ( 1.0 - post.pi ) / ( 1.0 - post.pi + post.pi * a );

        // Draw a new gamma
        std::bernoulli_distribution dist_bernoulli( 1.0 - b );
        post.gamma[s] = dist_bernoulli( state.rand_gen );
    }
}

// -------------------------------------------------------------------------
//     sample_and_update_beta
// -------------------------------------------------------------------------

void sample_and_update_beta(
    Data const& data, Hyperparams const& hyper, Posteriors& post, State& state
) {
    assert( data.num_indiv == data.x.rows() );
    assert( data.num_indiv == state.circle_product_x_beta.size() );
    assert( data.num_snps  == data.x.cols() );
    assert( data.num_snps  == post.beta.size() );
    assert( data.num_snps  == post.gamma.size() );

    // Update beta for all snps
    for( size_t s = 0; s < data.num_snps; ++s ) {
        // Remove the circ prod of the non-updated beta from current snp in our cache
        for( size_t i = 0; i < data.num_indiv; ++i ) {
            auto const circ_prod = data.x(i, s) * post.beta[s] + 1.0;
            state.circle_product_x_beta[i] /= circ_prod;
        }

        // Update the mean and variance using the current state.
        update_state_snp_cache( data, post, state, s );

        // Update the beta for the spike case.
        // In that case, we do not need to update circle_product_x_beta any more;
        // we removed the previous snp value, and do not want to add a new one.
        // If we have a non-zero gamma (slab case), we need to draw a new beta,
        // and so continue after this check.
        if( post.gamma[s] == 0 ) {
            post.beta[s] = 0.0;
            continue;
        }

        // Draw a new beta
        std::normal_distribution<> dist_normal(
            state.snp_mean[s], std::sqrt( state.snp_variance[s] )
        );
        post.beta[s] = dist_normal( state.rand_gen );

        // Ensure minimum beta threshold
        assert( hyper.min_beta > 0.0 );
        if( std::abs( post.beta[s] ) < hyper.min_beta ) {
           post.beta[s] = 0.0;
           continue;
        }

        // Add the updated value to current snp in our cache
        for( size_t i = 0; i < data.num_indiv; ++i ) {
            auto const circ_prod = data.x(i, s) * post.beta[s] + 1.0;
            state.circle_product_x_beta[i] *= circ_prod;
        }
    }

    // Finally, we need to recompute all caches to match with the whole lot of updated betas.
    refresh_state_cache_snp_values( data, post, state );
}

// =================================================================================================
//     Convergence
// =================================================================================================

// -------------------------------------------------------------------------
//     geweke_z_scores
// -------------------------------------------------------------------------

std::vector<double> geweke_z_scores(
    std::vector<double> const& x, double first = 0.1, double last = 0.5, size_t intervals = 20
) {
    // Input checks
    if(
        ! std::isfinite(first) || first <= 0.0 || first >= 1.0 ||
        ! std::isfinite(last)  || last  <= 0.0 || last  >= 1.0 ||
        first + last >= 1.0 || intervals == 0
    ) {
        throw std::invalid_argument(
            "Invalid intervals for Geweke convergence analysis: first=" + std::to_string( first ) +
            ", last=" + std::to_string( last ) + ", intervals=" + std::to_string( intervals )
        );
    }

    // Prepare and check boundary cases
    std::vector<double> zscores;
    if( x.size() < 2 ) {
        return zscores;
    }

    // Last index
    auto const end_idx = x.size() - 1;

    // Start intervals going up to the <last>% of the chain
    auto const last_start_idx = ( 1.0 - last ) * static_cast<double>( end_idx );

    // Loop over intervals
    size_t start_idx = 0;
    while( start_idx < static_cast<size_t>( last_start_idx )) {

        // Compute moments of first slice
        Moments first_slice_moments;
        auto const first_slice_end_idx = start_idx + static_cast<size_t>(first * (end_idx - start_idx));
        assert( start_idx < x.size() && first_slice_end_idx < x.size() );
        assert( start_idx <= first_slice_end_idx );
        for( size_t i = start_idx; i < first_slice_end_idx; ++i ) {
            first_slice_moments.push( x[i] );
        }

        // Compute moments of last slice
        Moments last_slice_moments;
        auto const last_slice_start_idx = end_idx - static_cast<size_t>(last * (end_idx - start_idx));
        assert( last_slice_start_idx < x.size() );
        for( size_t i = last_slice_start_idx; i < x.size(); ++i ) {
            last_slice_moments.push( x[i] );
        }

        // Compute z statistic
        auto const sqrt_var_sum = std::sqrt(
            first_slice_moments.variance() + last_slice_moments.variance()
        );
        auto const z = ( first_slice_moments.mean() - last_slice_moments.mean() ) / sqrt_var_sum;

        // Store result. The python implementation that we based this function on
        // also stores the start_idx here, but we do not need that for our case here, see
        // https://github.com/owlas/pymc3/blob/master/pymc3/diagnostics.py
        zscores.push_back( z );

        // Move to the next start index
        start_idx += static_cast<size_t>( last_start_idx / static_cast<double>( intervals - 1 ));
    }
    return zscores;
}

// -------------------------------------------------------------------------
//     update_zscore_with_trace_variable
// -------------------------------------------------------------------------

void update_zscore_with_trace_variable(
    Hyperparams const& hyper, Trace const& trace,
    std::function<double(TraceEntry const&)> select_value,
    double& max_zscore
) {
    // Get a vector of the variable along all traces.
    auto const trace_size = trace.entries.size();
    std::vector<double> values;
    values.reserve( trace_size );
    for( size_t t = 0; t < trace_size; ++t ) {
        values.push_back( select_value( trace.entries[t] ));
    }

    // Compute the zscore of the values, and if any is larger than the current one, store it.
    auto const zscores = geweke_z_scores(
        values, hyper.geweke_first, hyper.geweke_last, hyper.geweke_intervals
    );
    for( auto z : zscores ) {
        max_zscore = std::max( max_zscore, std::abs( z ));
    }
}

// -------------------------------------------------------------------------
//     has_converged
// -------------------------------------------------------------------------

bool has_converged( Data const& data, Hyperparams const& hyper, Trace const& trace )
{
    // We compute the z scores of all our posterior traces of interest.
    // Our trace stores the posterios etc in a way that is not easy to iterate over,
    // so we need to build vectors of the variables first. This is kind of expensive,
    // but we only run this function every 1000 or so iterations, so that's okay for now.
    // Might optimize later.
    double max_zscore = 0.0;

    // Test alpha convergence for all covariates
    for( size_t c = 0; c < data.num_covar; ++c ) {
        update_zscore_with_trace_variable(
            hyper, trace,
            [&]( TraceEntry const& entry ){
                assert( data.num_covar == entry.post.alpha.size() );
                return entry.post.alpha[c];
            },
            max_zscore
        );
    }

    // Test convergence of the top n beta values
    for( size_t b = 0; b < hyper.num_top_betas; ++b ) {
        update_zscore_with_trace_variable(
            hyper, trace,
            [&]( TraceEntry const& entry ){
                assert( hyper.num_top_betas == entry.top_betas.size() );
                return entry.top_betas[b];
            },
            max_zscore
        );
    }

    // Test convergence of large beta_ratio
    update_zscore_with_trace_variable(
        hyper, trace,
        [&]( TraceEntry const& entry ){
            return entry.stats.large_beta_ratio;
        },
        max_zscore
    );

    // Test convergence of sigma_1
    // update_zscore_with_trace_variable(
    //     hyper, trace,
    //     [&]( TraceEntry const& entry ){
    //        return entry.post.sigma_1;
    //     },
    //     max_zscore
    // );

    // Test convergence of sigma_e
    update_zscore_with_trace_variable(
        hyper, trace,
        [&]( TraceEntry const& entry ){
            return entry.post.sigma_e;
        },
        max_zscore
    );

    // Test convergence of total_heritability
    update_zscore_with_trace_variable(
        hyper, trace,
        [&]( TraceEntry const& entry ){
            return entry.stats.total_heritability;
        },
        max_zscore
    );

    // Finally, test if the largest zscore we found is good for convergence
    if( max_zscore < hyper.convergence_max_zscore ) {
        LOG_INFO << "Convergence has been reached";
        return true;
    }
    return false;
}

// =================================================================================================
//     Trace
// =================================================================================================

// -------------------------------------------------------------------------
//     push_to_trace
// -------------------------------------------------------------------------

void push_to_trace(
    size_t iteration, Posteriors const& post, Statistics const& stats,
    Hyperparams const& hyper, Trace& trace
) {
    // Prepare a trace entry with copies of the current samplign state
    TraceEntry entry;
    entry.iteration = iteration;
    entry.post = post;
    entry.stats = stats;

    // For the zscore test, we also want to use the top n absolute beta values of each iteration.
    // We use an efficient selection algorithm that does not need to sort the whole vector.
    // We get the result with the signs still there, but sorted by absolute value, largest first.
    auto abs_greater = []( double l, double r ){
        return std::abs( l ) > std::abs( r );
    };
    entry.top_betas = n_first_elements(
        post.beta.begin(), post.beta.end(), hyper.num_top_betas, abs_greater
    );

    // Move the entry to the end of the trace
    trace.entries.push_back( std::move( entry ));
}

// -------------------------------------------------------------------------
//     pop_from_trace
// -------------------------------------------------------------------------

void pop_from_trace( Trace& trace, size_t num_elements )
{
    num_elements = std::min( num_elements, trace.entries.size() );
    trace.entries.erase( trace.entries.begin(), trace.entries.begin() + num_elements );
}

// -------------------------------------------------------------------------
//     summarize_trace
// -------------------------------------------------------------------------

void summarize_trace( Data const& data, Trace const& trace, std::string const& file_prefix )
{
    // Prepare summary statistics of all posterior variables
    auto mom_alpha = std::vector<Moments>( data.num_covar );
    auto mom_beta  = std::vector<Moments>( data.num_snps );
    auto mom_gamma = std::vector<Moments>( data.num_snps );
    Moments mom_pi;
    Moments mom_sigma_1;
    Moments mom_sigma_e;
    Moments mom_poly;

    // Use the whole trace for the statistics
    for( auto const& entry : trace.entries ) {
        // Push alpha
        assert( data.num_covar == entry.post.alpha.size() );
        for( size_t c = 0; c < data.num_covar; ++c ) {
            mom_alpha[c].push( entry.post.alpha[c] );
        }

        // Push beta
        assert( data.num_snps == entry.post.beta.size() );
        for( size_t s = 0; s < data.num_snps; ++s ) {
            mom_beta[s].push( entry.post.beta[s] );
        }

        // Push gamma
        assert( data.num_snps == entry.post.gamma.size() );
        for( size_t s = 0; s < data.num_snps; ++s ) {
            mom_gamma[s].push( static_cast<double>( entry.post.gamma[s] ));
        }

        // Push others
        mom_pi.push( entry.post.pi );
        mom_sigma_1.push( entry.post.sigma_1 );
        mom_sigma_e.push( entry.post.sigma_e );
        mom_poly.push( entry.stats.polygenicity );
    }

    // Debug print of results
    LOG_INFO << "Trace summary:";
    LOG_INFO << "alpha";
    for( auto const& mom_a : mom_alpha ) {
        LOG_INFO << " - " << mom_a.mean() << " ± " << mom_a.stddev();
    }
    LOG_INFO << "beta";
    for( auto const& mom_b : mom_beta ) {
        LOG_INFO << " - " << mom_b.mean() << " ± " << mom_b.stddev();
    }
    LOG_INFO << "gamma";
    for( auto const& mom_g : mom_gamma ) {
        LOG_INFO << " - " << mom_g.mean() << " ± " << mom_g.stddev();
    }
    LOG_INFO << "pi           " << mom_pi.mean() << " ± " << mom_pi.stddev();
    LOG_INFO << "sigma_1      " << mom_sigma_1.mean() << " ± " << mom_sigma_1.stddev();
    LOG_INFO << "sigma_e      " << mom_sigma_e.mean() << " ± " << mom_sigma_e.stddev();
    LOG_INFO << "polygenicity " << mom_poly.mean() << " ± " << mom_poly.stddev();

    // Also write a trace summary file.
    auto summary_target = to_file( file_prefix + "_trace_summary.csv" );
    (*summary_target) << "value\tmean\tstddev\n";
    (*summary_target) << "pi\t" << mom_pi.mean() << "\t" << mom_pi.stddev() << "\n";
    (*summary_target) << "sigma_1\t" << mom_sigma_1.mean() << "\t" << mom_sigma_1.stddev() << "\n";
    (*summary_target) << "sigma_e\t" << mom_sigma_e.mean() << "\t" << mom_sigma_e.stddev() << "\n";
    (*summary_target) << "polygenicity\t" << mom_poly.mean() << "\t" << mom_poly.stddev() << "\n";
}

// -------------------------------------------------------------------------
//     write_trace
// -------------------------------------------------------------------------

void write_trace( Trace const& trace, std::string const& file_prefix )
{
    // Open file targets. Can make them compressed if we want.
    auto var_target   = to_file( file_prefix + "_trace_var.csv" );
    auto alpha_target = to_file( file_prefix + "_trace_alpha.csv" );
    auto beta_target  = to_file( file_prefix + "_trace_beta.csv" );
    auto gamma_target = to_file( file_prefix + "_trace_gamma.csv" );

    // Write headers for the var target, as that one has some mumbp jumbo columns.
    (*var_target) << "iteration\tpi\tsigma_1\tsigma_e\tpolygenicity\tgenetic_var\tpheno_var\t";
    (*var_target) << "large_beta_ratio\tlarge_beta_heritability\ttotal_heritability\n";

    // Now write the whole trace.
    for( auto const& entry : trace.entries ) {
        (*var_target) << entry.iteration << "\t";
        (*var_target) << entry.post.pi << "\t";
        (*var_target) << entry.post.sigma_1 << "\t";
        (*var_target) << entry.post.sigma_e << "\t";
        (*var_target) << entry.stats.polygenicity << "\t";
        (*var_target) << entry.stats.genetic_var << "\t";
        (*var_target) << entry.stats.pheno_var << "\t";
        (*var_target) << entry.stats.large_beta_ratio << "\t";
        (*var_target) << entry.stats.large_beta_heritability << "\t";
        (*var_target) << entry.stats.total_heritability << "\n";

        join( alpha_target->ostream(), entry.post.alpha, "\t" );
        join( beta_target->ostream(),  entry.post.beta, "\t" );
        join( gamma_target->ostream(), entry.post.gamma, "\t" );
        (*alpha_target) << "\n";
        (*beta_target) << "\n";
        (*gamma_target) << "\n";
    }
}

// =================================================================================================
//     Stats
// =================================================================================================

// -------------------------------------------------------------------------
//     compute_sample_statistics
// -------------------------------------------------------------------------

Statistics compute_sample_statistics(
    Data const& data, Hyperparams const& hyper, Posteriors const& post, State const& state
) {
    assert( data.num_indiv == data.x.rows() );
    assert( data.num_indiv == data.y.size() );
    assert( data.num_indiv == state.circle_product_x_beta.size() );
    assert( data.num_indiv == state.product_c_alpha.size() );
    assert( data.num_snps  == data.x.cols() );
    assert( data.num_snps  == post.beta.size() );
    assert( data.num_snps  == post.gamma.size() );
    assert( std::isfinite( hyper.large_beta ) && hyper.large_beta > 0.0 );

    // Return value
    Statistics stats;

    // We need to compute some variances, using a one pass algorithm for that
    Moments geno_moments;
    Moments pheno_moments;
    Moments large_beta_moments;

    // Get the large beta count, and sum up the polygenicity
    size_t large_beta_count = 0;
    for( size_t s = 0; s < data.num_snps; ++s ) {
        if( std::abs( post.beta[s] ) > hyper.large_beta ) {
            ++large_beta_count;
        }
        stats.polygenicity += post.gamma[s];
    }

    // Go through the individuals and compute the relevant variances and sums
    for( size_t i = 0; i < data.num_indiv; ++i ) {
        geno_moments.push( state.circle_product_x_beta[i] );
        pheno_moments.push( data.y[i] - state.product_c_alpha[i] );

        // Compute circle product for the current individual for its moment
        double circ_prod = 1.0;
        for( size_t s = 0; s < data.num_snps; ++s ) {
            if( std::abs( post.beta[s] ) > hyper.large_beta ) {
                circ_prod *= 1.0 + data.x(i, s) * post.beta[s];
            }
        }
        large_beta_moments.push( circ_prod );
    }

    // Compute our final statistics
    stats.genetic_var = geno_moments.variance();
    stats.pheno_var = pheno_moments.variance();
    stats.large_beta_ratio = static_cast<double>( large_beta_count ) / post.beta.size();
    stats.large_beta_heritability = large_beta_moments.variance() / stats.pheno_var;
    stats.total_heritability = stats.genetic_var / stats.pheno_var;
    return stats;
}

// -------------------------------------------------------------------------
//     valid_sample_statistics
// -------------------------------------------------------------------------

bool valid_sample_statistics( Statistics const& stats )
{
    // Previous alternative checks, not currently used.
    // bool const herit_too_large = stats.large_beta_heritability > 1.0;
    // bool const herit_larger_total = stats.large_beta_heritability > stats.total_heritability;

    // Check that we got a usable sample
    if( stats.total_heritability > 1 or stats.total_heritability < 0.01 ) {
        LOG_INFO << "unrealistic beta sample:";
        LOG_INFO << " - genetic_var             " << stats.genetic_var;
        LOG_INFO << " - pheno_var               " << stats.pheno_var;
        LOG_INFO << " - large_beta_ratio        " << stats.large_beta_ratio;
        LOG_INFO << " - large_beta_heritability " << stats.large_beta_heritability;
        LOG_INFO << " - total_heritability      " << stats.total_heritability;
        return false;
    }
    return true;
}

// =================================================================================================
//     Gibbs Sampler
// =================================================================================================

// -------------------------------------------------------------------------
//     initialize_hyperparams
// -------------------------------------------------------------------------

void initialize_hyperparams( Data const& data, Hyperparams& hyper )
{
    hyper.pi_b = data.num_snps / hyper.pi_b_ratio;
    if( hyper.pi_b <= 1.0 ) {
        throw std::runtime_error( "Too few SNPs, chain would not converge." );
    }
}

// -------------------------------------------------------------------------
//     initial_posteriors
// -------------------------------------------------------------------------

Posteriors initial_posteriors( Data const& data, Hyperparams const& hyper, State& state )
{
    Posteriors post;
    auto const num_snps  = data.x.cols();
    auto const num_covar = data.c.cols();

    // Init scalars randomly (needed for multi-threaded chains)
    assert( hyper.pi_b > 1.0 );
    post.pi = draw_pi( hyper.pi_a, hyper.pi_b, state.rand_gen );
    post.sigma_1 = draw_sigma_std_dev( hyper.sigma_1_a, hyper.sigma_1_b, state.rand_gen );
    post.sigma_e = draw_sigma_std_dev( hyper.sigma_e_a, hyper.sigma_e_b, state.rand_gen );
    LOG_DBG << "Initial pi: " << post.pi;
    LOG_DBG << "Initial sigma_1: " << post.sigma_1;
    LOG_DBG << "Initial sigma_e: " << post.sigma_e;

    // Hard coded reasonable values, not used in favor of random init.
    // post.pi = 0.001;
    // post.sigma_1 = 1.0;
    // post.sigma_e = 1.0;

    // Init alpha
    post.alpha.resize( num_covar );
    std::uniform_real_distribution<> dist_uniform_0_1( 0.0, 1.0 );
    for( auto& v : post.alpha ) {
        v = dist_uniform_0_1( state.rand_gen );
    }

    // Init gamma
    post.gamma.resize( num_snps );
    // std::binomial_distribution<> dist_binomial( 1, post.pi );
    std::bernoulli_distribution dist_bernoulli( post.pi );
    for( auto& v : post.gamma ) {
        // v = dist_binomial( state.rand_gen );
        v = dist_bernoulli( state.rand_gen );
    }

    // Init beta
    post.beta.resize( num_snps );
    std::normal_distribution<> dist_norm{ 0.0, post.sigma_1 };
    for( size_t i = 0; i < num_snps; ++i ) {
        post.beta[i] = post.gamma[i] == 0 ? 0.0 : dist_norm( state.rand_gen );
    }

    return post;
}

// -------------------------------------------------------------------------
//     sampling_step
// -------------------------------------------------------------------------

void sampling_step(
    Data const& data, Hyperparams const& hyper, Posteriors& post, State& state
) {
    //assert( std::isfinite( hyper.min_sigma_1 ) && hyper.min_sigma_1 > 0.0 );
    post.sigma_1 = sample_sigma_1( hyper, post, state );
    // XING CHANGED; no need to do this condition anymore
    // if( post.sigma_1 < hyper.min_sigma_1 ) {
    //     post.sigma_1 = hyper.min_sigma_1;
    //     post.pi = 0.0;
    // } else {
    //     post.pi = sample_pi( hyper, post, state );
    // }
    post.pi = sample_pi( hyper, post, state );
    post.sigma_e = sample_sigma_e( data, hyper, state );
    sample_and_update_gamma( data, post, state );
    sample_and_update_alpha( data, post, state );
    sample_and_update_beta( data, hyper, post, state );
}

// -------------------------------------------------------------------------
//     sampling
// -------------------------------------------------------------------------

void sampling(
    Data const& data, Runparams const& run, Hyperparams const& hyper,
    Posteriors& post, State& state, Trace& trace
) {
    size_t it = 1;
    size_t max_it = run.burnin_iterations + run.regular_iterations;

    while( it <= max_it && it <= run.test_convergence_stop ) {
        LOG_TIME << "Iteration " << it;

        // Do all sampling updates. We need a copy of the posterior here,
        // so that we can discard it if it's not good.
        auto post_update = post;
        sampling_step( data, hyper, post_update, state );
        // LOG_TIME << "Sampling done";

        // Compute sanity metrics to see if this is good.
        auto const stats = compute_sample_statistics( data, hyper, post_update, state );
        if( it > run.sanity_iterations && ! valid_sample_statistics( stats )) {
            // If we want to discard this sample, we need to make sure that the cached values
            // are based on the previous data, instead of their current state that is based
            // on the to-be-discarded posteriors.
            refresh_state_cache( data, post, state );
            continue;
        }

        // We have a sample that we want to use. Update the stored posterior.
        post = std::move( post_update );

        // After the burn-in, we keep a trace of all samples
        if( it > run.burnin_iterations ) {
            push_to_trace( it, post, stats, hyper, trace );
        }

        // After a while, start testing for convergence
        if(
            it > run.burnin_iterations + run.test_convergence_start &&
            it % run.test_convergence_interval == 0
        ) {
            LOG_INFO << "Testing convergence";
            if( has_converged( data, hyper, trace )) {
                break;
            }

            // If we have not converged yet, we run more iterations.
            // We remove a batch from the trace to reduce memory.
            pop_from_trace( trace, run.test_convergence_interval );
            max_it += run.test_convergence_interval;
        }

        ++it;
    }
    if( it >= run.test_convergence_stop ) {
        LOG_INFO << "Chain has not converged after " << it << " iterations";
    }
}

// =================================================================================================
//     Main
// =================================================================================================

int main( int argc, char** argv )
{
    // Activate logging.
    utils::Logging::log_to_stdout();
    utils::Logging::details.time = true;
    LOG_INFO << "Started";

    // // Input files, hard coded for testing now
    // std::string const indir = "/home/lucas/Downloads/gibbs/";
    //
    // // std::string const x_file = indir + "large_X.txt";
    // // std::string const y_file = indir + "large_multiplicative_y.txt";
    // // std::string const c_file = indir + "large_C.txt";
    // // std::string const prefix = indir + "large";
    //
    // std::string const x_file = indir + "small_X.txt";
    // std::string const y_file = indir + "small_multiplicative_y.txt";
    // std::string const c_file = indir + "small_C.txt";
    // std::string const prefix = indir + "small";

    // Check if the command line contains the right number of arguments.
    if (argc != 5) {
        throw std::runtime_error(
            "Usage: " + std::string( argv[0] ) + " x.mat y.mat c.mat out_prefix\n"
        );
    }

    // Input files
    auto const x_file = std::string( argv[1] );
    auto const y_file = std::string( argv[2] );
    auto const c_file = std::string( argv[3] );
    auto const prefix = std::string( argv[4] );

    // Read the data
    auto const data = read_input_data( x_file, y_file, c_file );

    // Set up the environment
    State state;
    Runparams run;
    Hyperparams hyper;
    Trace trace;
    initialize_hyperparams( data, hyper );
    auto post = initial_posteriors( data, hyper, state );
    init_state_cache( data, post, state );
    trace.entries.reserve( run.test_convergence_start );

    // Start the chain
    sampling( data, run, hyper, post, state, trace );

    // Summarize and write the trace
    summarize_trace( data, trace, prefix );
    write_trace( trace, prefix );

    LOG_INFO << "Finished";
    return 0;
}
