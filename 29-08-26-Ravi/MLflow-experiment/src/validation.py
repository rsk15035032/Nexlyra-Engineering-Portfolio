import pandas as pd


def validate_dataset(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):

    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    print("\nShapes")

    print("X_train:", X_train.shape)
    print("X_val  :", X_val.shape)
    print("X_test :", X_test.shape)

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print("\nTarget distribution")

    print(
        "Train:"
    )

    print(
        y_train.value_counts(
            normalize=True
        )
    )

    print(
        "\nValidation:"
    )

    print(
        y_val.value_counts(
            normalize=True
        )
    )

    print(
        "\nTest:"
    )

    print(
        y_test.value_counts(
            normalize=True
        )
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values")

    missing = (
        X_train.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        missing[
            missing > 0
        ].head(20)
    )

    # --------------------------------------------------------
    # Duplicate columns
    # --------------------------------------------------------

    duplicate_columns = (
        X_train.columns[
            X_train.columns.duplicated()
        ]
    )

    print(
        "\nDuplicate columns:",
        len(duplicate_columns)
    )

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    print(
        "\nUnique target values:",
        y_train.unique()
    )

    print("\nValidation completed.")