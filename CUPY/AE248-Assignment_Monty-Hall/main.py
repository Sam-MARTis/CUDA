import cupy
iterations = 100
game_size = 2**28
s = 0
# a_gpu = cupy.empty((game_size), dtype= cupy.int8)
for i in range(iterations):
    a_gpu = cupy.random.randint(0, 3, size=(game_size), dtype=cupy.int8)
    s += cupy.sum(a_gpu==0)
print(s/(iterations*game_size))
# print(cupy.sum(b)/game_size)


