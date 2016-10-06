
array = [3, 6, 1, 14]
lowInt = 0
highInt = 3


def quicksort (array , lowInt, highInt):
	if lowInt < highInt:
		pivot_location = partition(array, lowInt, highInt)
		quicksort(array, lowInt, pivot_location-1)
		quicksort(array, pivot_location+1, highInt)
	return array


def partition (array, lowInt, highInt):
	pivot = array[lowInt]
	leftwall = lowInt

	for i in range(lowInt+1, highInt):
		if array[i]<pivot:
			leftwall = leftwall+1
			array[i], array[leftwall] = array[leftwall], array[i]
	array[lowInt], array[leftwall] = array[leftwall], array[lowInt]

	return leftwall  


print quicksort(array, lowInt, highInt)