# load required libraries
import numpy as np

# define a function for bootstrap simulation
def bootstrap_from_fractions(
    team_name,
    t_fractions,
    h_fractions,
    t_total,
    h_total,
    f_value,
    n_iterations=10000,
    seed=42,
):
    
# set seed
    np.random.seed(seed)
 
# normalize within each array so they sum to 1
    t_fractions = np.array(t_fractions, dtype=float)
    h_fractions = np.array(h_fractions, dtype=float)

# create empty lists for tracking simulations
    t_fractions /= t_fractions.sum()
    h_fractions /= h_fractions.sum()
 
# point values aligned to state index order
 # order = [S2=2, S3=3, TO=0, E=0, F=f_value]
    point_values = np.array([2.0, 3.0, 0.0, 0.0, f_value])
 
 # create empty lists for tracking simulations
    t_ortg_sims = np.empty(n_iterations)
    h_ortg_sims = np.empty(n_iterations)
 
 # simulation using a for loop
    for i in range(n_iterations):
        # use multinomial distribution to simulate outcomes
        t_sim = np.random.multinomial(t_total, t_fractions)
        h_sim = np.random.multinomial(h_total, h_fractions)
 
        # extract each count by index before multiplying
        t_s2, t_s3, t_to, t_e, t_f = (
            int(t_sim[0]), int(t_sim[1]),
            int(t_sim[2]), int(t_sim[3]), int(t_sim[4])
        )
        h_s2, h_s3, h_to, h_e, h_f = (
            int(h_sim[0]), int(h_sim[1]),
            int(h_sim[2]), int(h_sim[3]), int(h_sim[4])
        )
 
        # calculate expected value by index
        t_ev = (t_s2 * 2.0  +  t_s3 * 3.0  +  t_f * f_value) / t_total
        h_ev = (h_s2 * 2.0  +  h_s3 * 3.0  +  h_f * f_value) / h_total
 
        # convert to Offensive Rating and append to lists
        t_ortg_sims[i] = float(t_ev * 100)
        h_ortg_sims[i] = float(h_ev * 100)
 
    #95% confidence intervals 
    t_ci_lower, t_ci_upper = np.percentile(t_ortg_sims, [2.5, 97.5])
    h_ci_lower, h_ci_upper = np.percentile(h_ortg_sims, [2.5, 97.5])
 
    # calculate actual point estimate ORtg
    t_ortg_base = float(np.dot(t_fractions, point_values) * 100)
    h_ortg_base = float(np.dot(h_fractions, point_values) * 100)
 
    # formatted output 
    print(f"--- {team_name} Results ---")
    print(f"Transition ORtg: {t_ortg_base:.1f} [{t_ci_lower:.1f}, {t_ci_upper:.1f}]")
    print(f"Half-Court ORtg: {h_ortg_base:.1f} [{h_ci_lower:.1f}, {h_ci_upper:.1f}]")
    print(f"Difference:      {(t_ortg_base - h_ortg_base):+.1f}\n")
 
    return {
        "t_ortg": t_ortg_base, "t_ci": (t_ci_lower, t_ci_upper),
        "h_ortg": h_ortg_base, "h_ci": (h_ci_lower, h_ci_upper),
    }
 
 ### ── example ─────────────────────────────────────────────────────────────────

# define state outcome fractions [S2, S3, TO, E, F]
team_t_fracs = [55/159, 17/159, 45/159, 35/159, 7/159]
team_h_fracs = [132/497, 61/497, 156/497, 136/497, 12/497]
 
# run simulation
bootstrap_from_fractions(
    team_name="Howard",
    t_fractions=team_t_fracs,
    h_fractions=team_h_fracs,
    t_total=159,
    h_total=497,
    f_value=1.46,
)
 