dim = 50
cheack_dim = dim
it = 0
for i in range(3):
    it +=1
    print(it)
    cheack_dim /= 2
    print(round(cheack_dim))
    if round(cheack_dim) < 3:
        print(f'error -> {dim} : {round(cheack_dim)}')