from train_encoder import train_encoder
from precompute_vectors import precompute_vectors
from train_prediction import train_prediction
from visualize import watch_reconstruction, watch_prediction

# the menu: each entry maps a typed choice to the function that does that work.
# to add a future stage, write its function and add one line here.
OPERATIONS = {
    "1": ("Train abstraction model (autoencoder)", train_encoder),
    "2": ("Precompute V vectors from trained encoder", precompute_vectors),
    "3": ("Train prediction module (inverse + memory + prediction)", train_prediction),
    "4": ("Watch abstraction reconstructions", watch_reconstruction),
    "5": ("Watch predictions vs actual", watch_prediction),
}


def main():
    while True:
        print("\n=== Behavior Learning Agent ===")
        for key, (label, _) in OPERATIONS.items():
            print(f"  {key}. {label}")
        print("  q. quit")

        choice = input("choose an operation: ").strip()

        if choice == "q":
            print("bye")
            break

        if choice in OPERATIONS:
            label, func = OPERATIONS[choice]
            print(f"\n--- running: {label} ---")
            func()
        else:
            print("unknown choice, try again")


if __name__ == "__main__":
    main()