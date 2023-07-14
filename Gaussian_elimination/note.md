run respectively
        ./exercise_1 < ../ans/6.in > ../ans/r_6.ans
        ../ans/checker_run ../ans/6.ans ../ans/r_6.ans


all at the same time
        cd build
        chmod -R 777 .
        ./all_checker.sh

# Affine space
- No vector has a fixed origin like linear space
- No vector can be uniquely associated to a point
- No point which is distinguished as origin
- Any vector space may be viewed as an affine space
- In this case, elements of the vector space may be viewed as points of the affine space, or translation (vector)
- If we consider as a point, the zero vector == origin
- Adding a fixed vector to the elements of a linear subspace of a vector space produces an affine subspace
- One commonly say that this affine subspace has been obtained by translating the linear subspace by translating vector.
- Affine subspace has been obtained by translating but they don't contain origin of the linear space.
**- Two subspaces that share the same direction are said to be parallel.**
- This implies the following generalization of Playfair's axiom: Given direction V, for any point a of A there is one and only one affine subspace of direction V, which passes through a, namely the subspace a+V

# Ex2
- We need to find any point a such that laying on the shared direction by two subspaces;
- If they have common place the 
