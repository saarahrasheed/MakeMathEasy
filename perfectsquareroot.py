# Find the root of a perfect square

def perfect_root(number):
	num = 2
	while num*num <= number:
		if num == number/num:
			return num
		else:
			num += 1 
	return print(number, ' is not a perfect square.') 
