import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from hmm_models import GaussianHMMWrapper
from regime_assignment import solve_regime_assignment, calculate_regime_distance

# Set random seed for reproducibility
np.random.seed(42)

def test_synthetic_data_consistency():
    """
    Test regime consistency using synthetic data with clearly defined regimes.
    Tests if regimes maintain their identity when refitted.
    """
    print("\n" + "="*80)
    print("TEST 1: SYNTHETIC DATA CONSISTENCY")
    print("="*80)
    
    # Create synthetic data with clearly separated regimes
    regime1_data = np.random.multivariate_normal(
        mean=[1, 1], 
        cov=[[0.1, 0], [0, 0.1]], 
        size=100
    )
    regime2_data = np.random.multivariate_normal(
        mean=[5, 5], 
        cov=[[0.2, 0], [0, 0.2]], 
        size=100
    )
    
    # Combine into single dataset with clear regime structure
    data1 = pd.DataFrame(
        np.vstack([regime1_data, regime2_data]), 
        columns=['x', 'y'],
        index=pd.date_range(start='2023-01-01', periods=200, freq='D')
    )
    
    # Fit initial model
    print("Fitting initial model...")
    model1 = GaussianHMMWrapper(n_regimes=2)
    model1.fit(data1)
    states1 = model1.transform(data1)
    
    print("Model 1 means:")
    print(model1.model.means_.round(3))
    
    # Create slightly perturbed version of the same data
    regime1_data2 = np.random.multivariate_normal(
        mean=[1.1, 0.9], 
        cov=[[0.11, 0], [0, 0.09]], 
        size=100
    )
    regime2_data2 = np.random.multivariate_normal(
        mean=[5.1, 4.9], 
        cov=[[0.21, 0], [0, 0.19]], 
        size=100
    )
    
    data2 = pd.DataFrame(
        np.vstack([regime1_data2, regime2_data2]), 
        columns=['x', 'y'],
        index=pd.date_range(start='2023-01-01', periods=200, freq='D')
    )
    
    # Fit second model, passing previous model for consistent assignment
    print("\nFitting second model with reference to first model...")
    model2 = GaussianHMMWrapper(n_regimes=2)
    model2.fit(data2, previous_model=model1)
    states2 = model2.transform(data2)
    
    print("Model 2 means:")
    print(model2.model.means_.round(3))
    
    # Check regime consistency
    # Both datasets should have same regime structure: first 100 points in one regime, next 100 in other
    # The regime labels should be consistent across models
    
    # Get first regime label in first 100 points of first model
    regime1_label = states1.iloc[0:100].mode()[0]
    # Get first regime label in second 100 points of first model
    regime2_label = states1.iloc[100:200].mode()[0]
    
    # Check if same labels appear in corresponding segments of second model
    consistent_labeling_1 = (states2.iloc[0:100].mode()[0] == regime1_label)
    consistent_labeling_2 = (states2.iloc[100:200].mode()[0] == regime2_label)
    consistent_labeling = consistent_labeling_1 and consistent_labeling_2
    
    print(f"\nFirst model regimes: {regime1_label} (first 100), {regime2_label} (second 100)")
    print(f"Second model modes: {states2.iloc[0:100].mode()[0]} (first 100), {states2.iloc[100:200].mode()[0]} (second 100)")
    print(f"Consistent regime labeling: {consistent_labeling}")
    
    # Check index remap for second model
    print(f"\nSecond model index remapping: {model2.index_remap}")
    
    # Visualize the data and regimes
    plt.figure(figsize=(15, 6))
    
    # Plot for Model 1
    plt.subplot(1, 2, 1)
    for state in states1.unique():
        mask = states1 == state
        plt.scatter(data1.loc[mask, 'x'], data1.loc[mask, 'y'], 
                   label=f'Model 1: {state}', alpha=0.7)
    plt.title("Model 1 Regime Assignment")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot for Model 2
    plt.subplot(1, 2, 2)
    for state in states2.unique():
        mask = states2 == state
        plt.scatter(data2.loc[mask, 'x'], data2.loc[mask, 'y'], 
                   label=f'Model 2: {state}', alpha=0.7)
    plt.title("Model 2 Regime Assignment")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('synthetic_data_test.png')
    
    # Create confusion matrix between models
    states1_numeric = states1.str.replace('regime', '').astype(int)
    states2_numeric = states2.str.replace('regime', '').astype(int)
    
    confusion = pd.crosstab(
        states1_numeric, 
        states2_numeric, 
        rownames=['Model 1'], 
        colnames=['Model 2'],
        normalize='index'
    )
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(confusion, annot=True, cmap='Blues', fmt='.2%')
    plt.title('Regime Assignment Consistency Between Models')
    plt.savefig('regime_confusion_matrix.png')
    
    if not consistent_labeling:
        print("\nFAILED: Regime assignments not consistent between models!")
    else:
        print("\nPASSED: Regime assignments consistent between models!")
    
    return consistent_labeling


def test_assignment_matrix_properties():
    """
    Validates the mathematical properties of the assignment algorithm
    using a controlled cost matrix.
    """
    print("\n" + "="*80)
    print("TEST 2: ASSIGNMENT MATRIX PROPERTIES")
    print("="*80)
    
    # Create synthetic cost matrix
    cost_matrix = np.array([
        [0.1, 2.0, 3.0],  # Previous regime 0 is closest to new regime 0
        [2.5, 0.2, 1.8],  # Previous regime 1 is closest to new regime 1
        [3.0, 1.7, 0.3]   # Previous regime 2 is closest to new regime 2
    ])
    
    print("Cost matrix:")
    print(cost_matrix.round(3))
    
    # Get assignments using the actual assignment function
    print("\nRunning assignment algorithm...")
    assignments, cost = solve_regime_assignment(cost_matrix)
    
    # Convert assignment dict to matrix for validation
    assignment_matrix = np.zeros_like(cost_matrix)
    for prev_idx, curr_idx in assignments.items():
        assignment_matrix[prev_idx, curr_idx] = 1
    
    print("\nResulting assignment matrix:")
    print(assignment_matrix.astype(int))
    
    # Check properties:
    # 1. Each previous regime must be assigned exactly once (row sums = 1)
    row_sums = assignment_matrix.sum(axis=1)
    rows_valid = np.all(row_sums == 1)
    
    # 2. Each new regime gets at most one label (column sums ≤ 1)
    col_sums = assignment_matrix.sum(axis=0)
    cols_valid = np.all(col_sums <= 1)
    
    # 3. Optimal assignment check - should select minimum cost assignments
    # Calculate cost of the assignment
    assignment_cost = np.sum(assignment_matrix * cost_matrix)
    # Check if it's the diagonal (which is optimal in this case)
    expected_optimal_cost = cost_matrix[0,0] + cost_matrix[1,1] + cost_matrix[2,2]
    cost_optimal = np.isclose(assignment_cost, expected_optimal_cost)
    
    print(f"\nRow constraint satisfied (each row sums to 1): {rows_valid}")
    print(f"Row sums: {row_sums}")
    
    print(f"\nColumn constraint satisfied (each column sums to ≤1): {cols_valid}")
    print(f"Column sums: {col_sums}")
    
    print(f"\nAssignment cost: {assignment_cost:.3f}")
    print(f"Expected optimal cost: {expected_optimal_cost:.3f}")
    print(f"Assignment cost optimal: {cost_optimal}")
    
    # Visualize cost matrix and assignment
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cost_matrix, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Cost Matrix')
    plt.xlabel('New Regime Index')
    plt.ylabel('Previous Regime Index')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(assignment_matrix.astype(int), annot=True, cmap='Blues', fmt='d')
    plt.title('Assignment Matrix')
    plt.xlabel('New Regime Index')
    plt.ylabel('Previous Regime Index')
    
    plt.tight_layout()
    plt.savefig('assignment_matrices.png')
    
    test_passed = rows_valid and cols_valid and cost_optimal
    
    if test_passed:
        print("\nPASSED: Assignment matrix has all required properties!")
    else:
        print("\nFAILED: Assignment matrix does not satisfy all constraints!")
    
    return test_passed


def test_changing_regimes():
    """
    Tests if the assignment algorithm correctly handles regimes appearing
    and disappearing between fits.
    """
    print("\n" + "="*80)
    print("TEST 3: HANDLING REGIME CHANGES")
    print("="*80)
    
    # First dataset with 2 regimes
    regime1_data = np.random.multivariate_normal(
        mean=[1, 1], 
        cov=[[0.1, 0], [0, 0.1]], 
        size=100
    )
    regime2_data = np.random.multivariate_normal(
        mean=[5, 5], 
        cov=[[0.2, 0], [0, 0.2]], 
        size=100
    )
    
    # Combine into single dataset with clear regime structure
    data1 = pd.DataFrame(
        np.vstack([regime1_data, regime2_data]), 
        columns=['x', 'y'],
        index=pd.date_range(start='2023-01-01', periods=200, freq='D')
    )
    
    # Fit initial model with 2 regimes
    print("Fitting initial model with 2 regimes...")
    model1 = GaussianHMMWrapper(n_regimes=2)
    model1.fit(data1)
    states1 = model1.transform(data1)
    
    print("Model 1 means:")
    print(model1.model.means_.round(3))
    
    # Second dataset with 3 regimes (added a new one)
    regime1_data2 = np.random.multivariate_normal(
        mean=[1.1, 0.9], 
        cov=[[0.11, 0], [0, 0.09]], 
        size=100
    )
    regime2_data2 = np.random.multivariate_normal(
        mean=[5.1, 4.9], 
        cov=[[0.21, 0], [0, 0.19]], 
        size=100
    )
    regime3_data2 = np.random.multivariate_normal(
        mean=[10, 10], 
        cov=[[0.3, 0], [0, 0.3]], 
        size=100
    )
    
    data2 = pd.DataFrame(
        np.vstack([regime1_data2, regime2_data2, regime3_data2]), 
        columns=['x', 'y'],
        index=pd.date_range(start='2023-01-01', periods=300, freq='D')
    )
    
    # Fit second model with 3 regimes, passing previous model for consistent assignment
    print("\nFitting second model with 3 regimes (new regime appeared)...")
    model2 = GaussianHMMWrapper(n_regimes=3)
    model2.fit(data2, previous_model=model1)
    states2 = model2.transform(data2)
    
    print("Model 2 means:")
    print(model2.model.means_.round(3))
    
    # Get first regime label in first 100 points of first model
    regime1_label = states1.iloc[0:100].mode()[0]
    # Get first regime label in second 100 points of first model
    regime2_label = states1.iloc[100:200].mode()[0]
    
    # Check if same labels appear in corresponding segments of second model
    consistent_labeling_1 = (states2.iloc[0:100].mode()[0] == regime1_label)
    consistent_labeling_2 = (states2.iloc[100:200].mode()[0] == regime2_label)
    
    # New regime should be labeled as regime3
    new_regime_label = states2.iloc[200:300].mode()[0]
    new_regime_correct = (new_regime_label == "regime3")
    
    test_passed = consistent_labeling_1 and consistent_labeling_2 and new_regime_correct
    
    print(f"\nFirst model regimes: {regime1_label} (first 100), {regime2_label} (second 100)")
    print(f"Second model corresponding segments: {states2.iloc[0:100].mode()[0]} (first 100), {states2.iloc[100:200].mode()[0]} (second 100)")
    print(f"New regime label: {new_regime_label} (should be regime3)")
    
    print(f"\nConsistent labeling for pre-existing regimes: {consistent_labeling_1 and consistent_labeling_2}")
    print(f"Correct labeling for new regime: {new_regime_correct}")
    
    # Check index remap for second model
    print(f"\nSecond model index remapping: {model2.index_remap}")
    
    # Visualize
    plt.figure(figsize=(15, 10))
    
    # Plot for Model 1
    plt.subplot(2, 1, 1)
    for state in states1.unique():
        mask = states1 == state
        plt.scatter(data1.loc[mask, 'x'], data1.loc[mask, 'y'], 
                   label=f'Model 1: {state}', alpha=0.7)
    plt.title("Model 1 Regime Assignment (2 regimes)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot for Model 2
    plt.subplot(2, 1, 2)
    for state in states2.unique():
        mask = states2 == state
        plt.scatter(data2.loc[mask, 'x'], data2.loc[mask, 'y'], 
                   label=f'Model 2: {state}', alpha=0.7)
    plt.title("Model 2 Regime Assignment (3 regimes, new regime appeared)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regime_change_test.png')
    
    # Test regime disappearance - create a dataset with only two of the three regimes
    regime1_data3 = np.random.multivariate_normal(
        mean=[1.2, 0.8], 
        cov=[[0.12, 0], [0, 0.08]], 
        size=100
    )
    # Skip regime 2 to test disappearance
    regime3_data3 = np.random.multivariate_normal(
        mean=[10.1, 9.9], 
        cov=[[0.31, 0], [0, 0.29]], 
        size=100
    )
    
    data3 = pd.DataFrame(
        np.vstack([regime1_data3, regime3_data3]), 
        columns=['x', 'y'],
        index=pd.date_range(start='2023-01-01', periods=200, freq='D')
    )
    
    # Fit third model with 2 regimes, passing model2 for consistent assignment
    print("\nFitting third model with 2 regimes (regime disappeared)...")
    model3 = GaussianHMMWrapper(n_regimes=2)
    model3.fit(data3, previous_model=model2)
    states3 = model3.transform(data3)
    
    print("Model 3 means:")
    print(model3.model.means_.round(3))
    
    # The first segment should still be regime1 and the second should be regime3
    regime1_consistent = (states3.iloc[0:100].mode()[0] == regime1_label)
    regime3_consistent = (states3.iloc[100:200].mode()[0] == new_regime_label)
    
    print(f"\nThird model regimes: {states3.iloc[0:100].mode()[0]} (first 100), {states3.iloc[100:200].mode()[0]} (second 100)")
    print(f"Consistent labeling in model 3: regime1 consistent = {regime1_consistent}, regime3 consistent = {regime3_consistent}")
    
    # Check index remap for third model
    print(f"\nThird model index remapping: {model3.index_remap}")
    
    # Visualize model 3
    plt.figure(figsize=(10, 6))
    for state in states3.unique():
        mask = states3 == state
        plt.scatter(data3.loc[mask, 'x'], data3.loc[mask, 'y'], 
                   label=f'Model 3: {state}', alpha=0.7)
    plt.title("Model 3 Regime Assignment (2 regimes, middle regime disappeared)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regime_disappearance_test.png')
    
    disappearance_test_passed = regime1_consistent and regime3_consistent
    
    if test_passed and disappearance_test_passed:
        print("\nPASSED: Algorithm correctly handles both regime appearance and disappearance!")
    else:
        print("\nFAILED: Issues with handling regime changes!")
    
    return test_passed and disappearance_test_passed


def test_assignment_with_alpha_variations():
    """
    Tests how different values of alpha (weight between mean vs covariance)
    affect regime assignments.
    """
    print("\n" + "="*80)
    print("TEST 4: ALPHA PARAMETER SENSITIVITY")
    print("="*80)
    
    # Create regimes with different mean/covariance properties
    # Regime 1: Low mean, low covariance
    # Regime 2: High mean, high covariance
    # Regime 3: Low mean, high covariance
    # Regime 4: High mean, low covariance
    
    regime1 = (np.array([1, 1]), np.array([[0.1, 0], [0, 0.1]]))  # Low mean, low cov
    regime2 = (np.array([5, 5]), np.array([[0.5, 0], [0, 0.5]]))  # High mean, high cov
    regime3 = (np.array([1, 5]), np.array([[0.5, 0], [0, 0.1]]))  # Mixed mean, mixed cov
    regime4 = (np.array([5, 1]), np.array([[0.1, 0], [0, 0.5]]))  # Mixed mean, mixed cov
    
    # Perturb slightly for "new" regimes
    regime1_new = (np.array([1.1, 0.9]), np.array([[0.12, 0], [0, 0.08]]))
    regime2_new = (np.array([5.1, 4.9]), np.array([[0.52, 0], [0, 0.48]]))
    regime3_new = (np.array([0.9, 5.1]), np.array([[0.48, 0], [0, 0.12]]))
    regime4_new = (np.array([5.1, 0.9]), np.array([[0.08, 0], [0, 0.52]]))
    
    # Alpha values to test
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Store results
    results = []
    
    for alpha in alphas:
        print(f"\nTesting with alpha = {alpha}")
        
        # Create cost matrix
        cost_matrix = np.zeros((4, 4))
        
        for i, old_regime in enumerate([regime1, regime2, regime3, regime4]):
            for j, new_regime in enumerate([regime1_new, regime2_new, regime3_new, regime4_new]):
                cost_matrix[i, j] = calculate_regime_distance(
                    old_regime[0], old_regime[1],
                    new_regime[0], new_regime[1],
                    alpha=alpha
                )
        
        print("Cost matrix:")
        print(cost_matrix.round(3))
        
        # Get assignments
        assignments, _ = solve_regime_assignment(cost_matrix)
        
        # Convert to assignment matrix
        assignment_matrix = np.zeros_like(cost_matrix)
        for prev_idx, curr_idx in assignments.items():
            assignment_matrix[prev_idx, curr_idx] = 1
        
        print("Assignment matrix:")
        print(assignment_matrix.astype(int))
        
        # Check for diagonal assignment (perfect matching)
        diagonal_match = np.array_equal(assignment_matrix, np.eye(4))
        print(f"Perfect diagonal assignment: {diagonal_match}")
        
        results.append({
            'alpha': alpha,
            'cost_matrix': cost_matrix,
            'assignment_matrix': assignment_matrix,
            'diagonal_match': diagonal_match
        })
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    for i, result in enumerate(results):
        plt.subplot(len(alphas), 2, 2*i+1)
        sns.heatmap(result['cost_matrix'], annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title(f'Cost Matrix (alpha={result["alpha"]})')
        plt.xlabel('New Regime Index')
        plt.ylabel('Previous Regime Index')
        
        plt.subplot(len(alphas), 2, 2*i+2)
        sns.heatmap(result['assignment_matrix'].astype(int), annot=True, cmap='Blues', fmt='d')
        plt.title(f'Assignment Matrix (alpha={result["alpha"]})')
        plt.xlabel('New Regime Index')
        plt.ylabel('Previous Regime Index')
    
    plt.tight_layout()
    plt.savefig('alpha_sensitivity.png')
    
    # Check if we have consistent assignments for middle alpha values
    consistent_middle = results[2]['diagonal_match']  # alpha=0.5
    
    if consistent_middle:
        print("\nPASSED: Alpha=0.5 gives diagonal assignment as expected!")
    else:
        print("\nFAILED: Alpha=0.5 does not give diagonal assignment!")
    
    # Show sensitivity of alpha
    print("\nAlpha sensitivity analysis:")
    for result in results:
        print(f"Alpha={result['alpha']}: diagonal match = {result['diagonal_match']}")
    
    return consistent_middle


def main():
    """
    Main function to run all tests and report overall results.
    """
    print("\n" + "="*80)
    print(" GAUSSIAN HMM REGIME ASSIGNMENT TEST SUITE ")
    print("="*80)
    
    # Run all tests
    test1 = test_synthetic_data_consistency()
    test2 = test_assignment_matrix_properties()
    test3 = test_changing_regimes()
    test4 = test_assignment_with_alpha_variations()
    
    # Summarize results
    print("\n" + "="*80)
    print(" TEST RESULTS SUMMARY ")
    print("="*80)
    print(f"Test 1 (Synthetic Data Consistency): {'PASSED' if test1 else 'FAILED'}")
    print(f"Test 2 (Assignment Matrix Properties): {'PASSED' if test2 else 'FAILED'}")
    print(f"Test 3 (Handling Regime Changes): {'PASSED' if test3 else 'FAILED'}")
    print(f"Test 4 (Alpha Parameter Sensitivity): {'PASSED' if test4 else 'FAILED'}")
    
    all_passed = test1 and test2 and test3 and test4
    
    if all_passed:
        print("\nALL TESTS PASSED! The regime assignment implementation appears correct.")
    else:
        print("\nSome tests FAILED. Review the test output above for details.")
    
    print("\nVisualization files generated:")
    print("- synthetic_data_test.png - Basic consistency test visualization")
    print("- regime_confusion_matrix.png - Confusion matrix between fits")
    print("- assignment_matrices.png - Cost and assignment matrix visualization")
    print("- regime_change_test.png - Handling of new regimes")
    print("- regime_disappearance_test.png - Handling of disappeared regimes")
    print("- alpha_sensitivity.png - Sensitivity to alpha parameter")


if __name__ == "__main__":
    main()