import numpy as np


def calculate_regime_distance(mean_i, cov_i, mean_j, cov_j, alpha=0.5):
    """
    Calculate weighted distance between two Gaussian regimes.
    
    Parameters
    ----------
    mean_i : numpy.ndarray
        Mean vector of first regime, shape (n_features,)
    cov_i : numpy.ndarray
        Covariance matrix of first regime, shape (n_features, n_features) or (n_features,)
    mean_j : numpy.ndarray
        Mean vector of second regime, shape (n_features,)
    cov_j : numpy.ndarray
        Covariance matrix of second regime, shape (n_features, n_features) or (n_features,)
    alpha : float, default=0.5
        Weight for mean distance vs covariance distance (0 to 1)
    
    Returns
    -------
    float
        Weighted distance between regimes
    """
    # Ensure alpha is valid
    assert (alpha <= 1) and (alpha >= 0)
    
    # Convert diagonal covariance to full if needed
    if len(cov_i.shape) == 1:
        cov_i = np.diag(cov_i)
    if len(cov_j.shape) == 1:
        cov_j = np.diag(cov_j)
    
    # Calculate mean cost using norm-2 
    mean_cost = np.linalg.norm(mean_i - mean_j, ord=2)
    
    # Calculate covariance distance (Frobenius norm)
    cov_cost = np.linalg.norm(cov_i - cov_j, ord='fro')
    
    # Weighted combination 
    distance = alpha * mean_cost + (1 - alpha) * cov_cost
    
    return distance







def get_cost_matrix(past_regimes: dict, new_regimes: dict, alpha=0.5) -> np.ndarray:
    """
    Calculate costs between all pairs of label assignment
    
    Parameters
    ----------
    past_regimes : dict
        Dictionary of previous regimes with keys as labels and values as regime objects
    new_regimes : dict
        Dictionary of new regimes with keys as temporary labels and values as regime objects
    alpha : float, default=0.5
        Weight for mean distance vs covariance distance (0 to 1)
        
    Returns
    -------
    numpy.ndarray
        Cost matrix of shape (len(past_regimes), len(new_regimes))
    """
    cost_mat = 10000 * np.ones(shape=(len(past_regimes), len(new_regimes)))
    
    print(f"[DEBUG] Creating cost matrix of shape: {len(past_regimes)} previous regimes × {len(new_regimes)} new regimes")
    print(f"[DEBUG] Previous regime labels: {list(past_regimes.keys())}")
    print(f"[DEBUG] New regime temp labels: {list(new_regimes.keys())}")
    
    for i, (past_label, past_rv) in enumerate(past_regimes.items()):
        for j, (new_label, new_rv) in enumerate(new_regimes.items()):
            cost_mat[i, j] = calculate_regime_distance(
                past_rv.mean, past_rv.cov, new_rv.mean, new_rv.cov, alpha=alpha
            )
            print(f"[DEBUG] Cost of assigning {past_label} to {new_label}: {cost_mat[i, j]:.4f}")
            
    print(f"[DEBUG] Full cost matrix:\n{np.round(cost_mat, 4)}")
    
    return cost_mat




def validate_assignment_matrix(assignment_matrix):
    """
    Validate assignment matrix to ensure it satisfies constraints
    
    Parameters
    ----------
    assignment_matrix : numpy.ndarray
        Assignment matrix of shape (n_prev_regimes, n_current_regimes)
    """
    print("\n[VALIDATION] Starting validation of assignment matrix")
    
    # Check that each previous regime is assigned exactly once
    row_sums = assignment_matrix.sum(axis=1)
    print(f"[VALIDATION] Rows sum (should all be 1): {row_sums}")
    rows_ok = np.all(row_sums == 1)
    print(f"[VALIDATION] Row sums check: {'PASS' if rows_ok else 'FAIL'}")
    
    # Check that each new regime gets at most one label
    col_sums = assignment_matrix.sum(axis=0)
    print(f"[VALIDATION] Columns sum (should be <= 1): {col_sums}")
    cols_ok = np.all(col_sums <= 1)
    print(f"[VALIDATION] Column sums check: {'PASS' if cols_ok else 'FAIL'}")
    
    # Overall validation result
    all_passed = rows_ok and cols_ok
    print(f"[VALIDATION] Overall validation: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed






def solve_regime_assignment(cost_matrix):
    """
    Solve the regime assignment problem using optimization.
    
    Parameters
    ----------
    cost_matrix : numpy.ndarray
        Matrix of assignment costs, shape (n_prev_regimes, n_current_regimes)
    
    Returns
    -------
    tuple
        (assignments, total_cost)
        assignments: dict mapping from previous regime idx to current regime idx
        total_cost: scalar value of the optimal assignment cost
    """
    m, n = cost_matrix.shape  # m=prev regimes, n=current regimes
    print(f"\n[REGIME ASSIGNMENT] Cost matrix shape: {m} previous regimes × {n} current regimes")
    print(f"[REGIME ASSIGNMENT] Cost matrix:\n{cost_matrix.round(4)}")
    
    try:
        import cvxpy as cp
    except ImportError:
        print("[REGIME ASSIGNMENT] CVXPY not installed. Using greedy assignment instead.")
        return _greedy_assignment(cost_matrix)
    
    # Decision variables - using integer variables as in label_assignment.py
    x = cp.Variable((m, n), integer=True)
    
    # Objective function - using Frobenius norm / m as in label_assignment.py
    obj_func = cp.Minimize(cp.atoms.norm(cp.multiply(cost_matrix, x), 'fro') / m)
    
    # Constraints
    constraints = [
        cp.sum(x, axis=0) <= 1,  # at most one label is assigned to a new regime
        cp.sum(x, axis=1) == 1,  # each label of the old regimes is assigned to exactly one new regime
        x >= 0  # non-negativity
    ]
    
    # Create problem
    problem = cp.Problem(obj_func, constraints)
    
    # Debug prints for problem definition
    print(f"[DEBUG] Objective function: {obj_func}")
    print(f"[DEBUG] Constraints:")
    for i, c in enumerate(constraints):
        print(f"[DEBUG]   {i}: {c}")
    
    # Try different solvers in order of preference, starting with SCIP as used in label_assignment.py
    solvers_to_try = ['SCIP', 'GLPK_MI', 'ECOS_BB', 'CBC']
    solver_found = False
    result = None
    
    for solver_name in solvers_to_try:
        try:
            print(f"[DEBUG] Trying solver: {solver_name}")
            result = problem.solve(solver=solver_name)
            solver_found = True
            print(f"[REGIME ASSIGNMENT] Using {solver_name} solver")
            print(f"[REGIME ASSIGNMENT] Problem status: {problem.status}")
            break
        except Exception as e:
            print(f"[DEBUG] Solver {solver_name} failed with error: {str(e)}")
            continue
    
    # If no solver worked, use greedy assignment
    if not solver_found or problem.status not in ["optimal", "optimal_inaccurate"]:
        print("[REGIME ASSIGNMENT] No MILP solver available or problem not solved. Using greedy assignment.")
        return _greedy_assignment(cost_matrix)
    
    # Get assignment matrix and convert to integers
    assign_mat = x.value
    assign_mat = assign_mat.astype(int)
    
    print(f"[DEBUG] Raw assignment matrix solution:")
    print(assign_mat)
    
    # Validate assignment matrix
    validate_assignment_matrix(assign_mat)
    
    # Convert assignment matrix to dictionary
    assignments = {}
    for i in range(m):
        for j in range(n):
            if assign_mat[i, j] == 1:
                assignments[i] = j
                print(f"[DEBUG] Assigned previous regime {i} to current regime {j}")
    
    print(f"[REGIME ASSIGNMENT] Assignment complete")
    print(f"[REGIME ASSIGNMENT] Total assignments: {len(assignments)}/{m}")
    print(f"[REGIME ASSIGNMENT] Assignments: {assignments}")
    print(f"[REGIME ASSIGNMENT] Objective value: {result:.4f}")
    
    return assignments, result









def relabel_new_regimes(previous_regimes, new_regimes, assign_mat):
    """
    Relabel new regimes based on assignment matrix, matching label_assignment.py's approach
    
    Parameters
    ----------
    previous_regimes : dict
        Dictionary of previous regimes with keys as labels and values as regime objects
    new_regimes : dict
        Dictionary of new regimes with keys as temporary labels and values as regime objects
    assign_mat : numpy.ndarray
        Assignment matrix of shape (len(previous_regimes), len(new_regimes))
        
    Returns
    -------
    dict
        Dictionary of relabeled regimes
    """
    # Get previous regime labels
    previous_regime_labels = list(previous_regimes.keys())
    
    print(f"[DEBUG] Previous regime labels: {previous_regime_labels}")
    print(f"[DEBUG] Assignment matrix:\n{assign_mat}")
    
    # Initialize new regime labels
    new_regime_labels = [None for _ in range(len(new_regimes))]
    
    # Create labels for newly added regimes
    newly_added_regime_labels = [f'regime{idx}' for idx in range(len(previous_regimes)+1, len(new_regimes)+1)]
    print(f"[DEBUG] Labels for newly added regimes: {newly_added_regime_labels}")
    
    newly_added_count = 0
    for idx in range(len(new_regimes)):
        assigned_old_indices = np.where(assign_mat[:, idx]==1)[0]
        if assigned_old_indices.shape[0] == 0:
            # This is a newly added regime
            new_regime_labels[idx] = newly_added_regime_labels[newly_added_count]
            print(f"[DEBUG] Column {idx} has no assignment, assigning new label: {new_regime_labels[idx]}")
            newly_added_count += 1
        else:
            # This regime inherits a previous label
            i = assigned_old_indices[0]
            new_regime_labels[idx] = previous_regime_labels[i]
            print(f"[DEBUG] Column {idx} inherits label from row {i}: {new_regime_labels[idx]}")
    
    print(f"[DEBUG] Final new regime labels: {new_regime_labels}")
    
    # Create the relabeled regimes dictionary
    new_regimes_list = list(new_regimes.values())
    relabeled = {new_label: regime for new_label, regime in zip(new_regime_labels, new_regimes_list)}
    
    print(f"[DEBUG] Relabeled regimes keys: {list(relabeled.keys())}")
    return relabeled


def validate_assignment(previous_regimes, new_regimes, assignment_matrix, relabeled_regimes):
    """
    Exhaustively check if the assignment is correct
    
    Parameters
    ----------
    previous_regimes : dict
        Dictionary of previous regimes with keys as labels and values as regime objects
    new_regimes : dict
        Dictionary of new regimes with keys as temporary labels and values as regime objects
    assignment_matrix : numpy.ndarray
        Assignment matrix of shape (len(previous_regimes), len(new_regimes))
    relabeled_regimes : dict
        Dictionary of relabeled regimes
        
    Returns
    -------
    bool
        True if all checks pass, False otherwise
    """
    print("\n[VALIDATION] Starting exhaustive validation of assignment")
    
    # Check assignment matrix shape
    print(f"[VALIDATION] Assignment matrix shape: {assignment_matrix.shape}")
    expected_shape = (len(previous_regimes), len(new_regimes))
    shape_ok = assignment_matrix.shape == expected_shape
    print(f"[VALIDATION] Shape check: {'PASS' if shape_ok else 'FAIL'} - Expected {expected_shape}")
    
    # Check that each previous regime is assigned exactly once
    row_sums = assignment_matrix.sum(axis=1)
    print(f"[VALIDATION] Rows sum (should all be 1): {row_sums}")
    rows_ok = np.all(row_sums == 1)
    print(f"[VALIDATION] Row sums check: {'PASS' if rows_ok else 'FAIL'}")
    
    # Check that each new regime gets at most one label
    col_sums = assignment_matrix.sum(axis=0)
    print(f"[VALIDATION] Columns sum (should be <= 1): {col_sums}")
    cols_ok = np.all(col_sums <= 1)
    print(f"[VALIDATION] Column sums check: {'PASS' if cols_ok else 'FAIL'}")
    
    # Check that number of relabeled regimes matches number of new regimes
    size_ok = len(relabeled_regimes) == len(new_regimes)
    print(f"[VALIDATION] Relabeled size check: {'PASS' if size_ok else 'FAIL'} - Expected {len(new_regimes)}, got {len(relabeled_regimes)}")
    
    # Check that all previous labels are preserved or new ones are added correctly
    prev_labels = set(previous_regimes.keys())
    new_labels = set(relabeled_regimes.keys())
    
    # Count how many previous labels are used in the relabeled regimes
    used_prev_labels = set()
    for i, row in enumerate(assignment_matrix):
        for j, val in enumerate(row):
            if val == 1:
                used_prev_labels.add(list(previous_regimes.keys())[i])
    
    print(f"[VALIDATION] Previous labels: {prev_labels}")
    print(f"[VALIDATION] Used previous labels: {used_prev_labels}")
    print(f"[VALIDATION] New labels: {new_labels}")
    
    # Check if all used previous labels are in the new labels
    labels_preserved = used_prev_labels.issubset(new_labels)
    print(f"[VALIDATION] Labels preservation check: {'PASS' if labels_preserved else 'FAIL'}")
    
    # Check if any new labels were added and they follow the correct naming convention
    if len(new_labels) > len(used_prev_labels):
        new_added_labels = new_labels - used_prev_labels
        print(f"[VALIDATION] New labels added: {new_added_labels}")
        
        # Check if the new labels follow the correct convention
        expected_new_labels = set()
        for i in range(len(previous_regimes)+1, len(new_regimes)+1):
            expected_new_labels.add(f'regime{i}')
        
        new_labels_ok = new_added_labels.issubset(expected_new_labels)
        print(f"[VALIDATION] New labels convention check: {'PASS' if new_labels_ok else 'FAIL'}")
        if not new_labels_ok:
            print(f"[VALIDATION] Expected new labels: {expected_new_labels}")
            print(f"[VALIDATION] Actual new labels: {new_added_labels}")
    else:
        new_labels_ok = True
    
    # Overall validation result
    all_passed = shape_ok and rows_ok and cols_ok and size_ok and labels_preserved and new_labels_ok
    print(f"[VALIDATION] Overall validation: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed







def _greedy_assignment(cost_matrix):
    """
    Fallback greedy assignment method when CVXPY fails.
    Simply assigns each previous regime to the closest current regime.
    
    Parameters
    ----------
    cost_matrix : numpy.ndarray
        Matrix of assignment costs, shape (n_prev_regimes, n_curr_regimes)
    
    Returns
    -------
    tuple
        (assignments_dict, total_cost)
        assignments_dict: dict mapping from previous regime idx to current regime idx
        total_cost: scalar value of the optimal assignment cost
    """
    print("[DEBUG] Using greedy assignment fallback")
    assignments = {}
    assigned_cols = set()
    
    # Sort all (i,j) pairs by cost
    flat_costs = []
    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            flat_costs.append((cost_matrix[i, j], i, j))
    
    flat_costs.sort()  # Sort by cost
    print(f"[DEBUG] Top 10 lowest cost pairs: {flat_costs[:10]}")
    
    # Assign greedily
    for cost, i, j in flat_costs:
        if i not in assignments and j not in assigned_cols:
            print(f"[DEBUG] Greedily assigning previous regime {i} to current regime {j} with cost {cost:.4f}")
            assignments[i] = j
            assigned_cols.add(j)
            
        # Stop if all previous regimes are assigned
        if len(assignments) == cost_matrix.shape[0]:
            break
    
    print(f"[DEBUG] Final greedy assignments: {assignments}")
    
    # Calculate total cost using Frobenius norm / num_old_regimes as in label_assignment.py
    assign_mat = np.zeros(cost_matrix.shape)
    for i, j in assignments.items():
        assign_mat[i, j] = 1
    
    total_cost = np.linalg.norm(assign_mat * cost_matrix, 'fro') / cost_matrix.shape[0]
    print(f"[DEBUG] Total greedy assignment cost: {total_cost:.4f}")
    
    return assignments, total_cost




