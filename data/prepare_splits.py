import pandas as pd

def split_1():
    # Train and Val (Derm7pt)
    train_derm7pt = pd.read_csv("derm7pt_train.csv")
    val_derm7pt = pd.read_csv("derm7pt_validation.csv")
    test_derm7pt = pd.read_csv("derm7pt_test.csv")

    train_df = pd.concat([train_derm7pt, test_derm7pt])
    val_df = val_derm7pt

    # Test (PH2)
    train_ph2 = pd.read_csv("PH2_train.csv")
    val_ph2 = pd.read_csv("PH2_validation.csv")
    test_ph2 = pd.read_csv("PH2_test.csv")

    test_df = pd.concat([train_ph2, val_ph2, test_ph2])

    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    # Prepare splits (convert DF to CSV)
    train_df.to_csv("train_raw.csv", index=False)
    val_df.to_csv("val_raw.csv", index=False)
    test_df.to_csv("test_raw.csv", index=False)

def split_2():
    # Train and Val (PH2)
    train_ph2 = pd.read_csv("PH2_train.csv")
    val_ph2 = pd.read_csv("PH2_validation.csv")
    test_ph2 = pd.read_csv("PH2_test.csv")

    train_df = pd.concat([train_ph2, test_ph2])
    val_df = val_ph2

    # Test (Derm7pt)
    train_derm7pt = pd.read_csv("derm7pt_train_seg.csv")
    val_derm7pt = pd.read_csv("derm7pt_validation_seg.csv")
    test_derm7pt = pd.read_csv("derm7pt_test_seg.csv")

    test_df = pd.concat([train_derm7pt, val_derm7pt, test_derm7pt])

    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    # Prepare splits (convert DF to CSV)
    train_df.to_csv("train_split2.csv", index=False)
    val_df.to_csv("val_split2.csv", index=False)
    test_df.to_csv("test_split2.csv", index=False)


if __name__ == '__main__':
    split_2()

