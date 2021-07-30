# Copyright (c) 2021 Leonardo Banchi <leonardo.banchi@unifi.it> 
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

using LinearAlgebra, BlockDiagonals

" symplectic matrix "
omega(n::Int) = BlockDiagonal([[0 1.; -1. 0] for k=1:n])
omega(A::AbstractMatrix) = omega(Int(size(A,1)/2))

" arrange diagonal elements in a block form "
wdiag(D) = BlockDiagonal([ Diagonal([d,d]) for d in D])

" check if a matrix is symplectic "
issymplectic(S) = S*omega(S)*S'≈omega(S)

" generate a random symplectic matrix "
function rand_symplectic(n) 
	A = randn(2n,2n)
	exp(Hermitian(A)*omega(n))
end

" generate a random covariance matrix "
function rand_covmat(n,σ=1)
	v = 1/2 .+ σ*rand(n)
	S = rand_symplectic(n)
	Symmetric(S*wdiag(v)*S')
end

function _will_init(V)
	@assert	size(V,1) == size(V,2) && iseven(size(V,1))
	Int(size(V,1)/2), omega(V)
end

"""
	D,S = williamson(V)

	find the Williamson decomposition such that 
	S'*wdiag(D)*S ≈ V
"""
function williamson(V)
	n, Ω = _will_init(V)
	iv12 = inv(sqrt(Symmetric(V)))
	J = iv12*Ω*iv12
	f = schur((J-J')/2)
	T, Q = f.Schur, f.vectors
	P = BlockDiagonal([ sign(T[2k-1,2k]) > 0 ? 
					   [1. 0; 0 1.] : [0 1.; 1. 0] for k=1:n])
	D = inv.(diag(P*T*P,1)[1:2:2n])
	iS = wdiag(sqrt.(D))*P*Q'*iv12
	D, Ω*iS*Ω
end

"""
	D,S = williamsonNew(V)

	find the Williamson decomposition such that 
	S'*wdiag(D)*S ≈ V
	new algorithm.
"""
function williamsonNew(V)
	n, Ω = _will_init(V)
	λ = sort(real(eigvals(V*Ω*1im)))[n+1:2n]
	S = zeros(2n,2n)
    for m=1:n
	    denom = λ[m]*prod(λ[k]^2-λ[m]^2 for k=1:n if k!=m) 
        Vλ = complex.(V)-(1im*λ[m])*Ω
	    minorden(k) = det(view(Vλ,setdiff(1:2n,2m),setdiff(1:2n,k)))/denom
        fam = real(minorden(2m))
	    for k=1:n
            faeven = k==m ? sqrt(fam) : minorden(2k)/sqrt(fam)
            faodd = -minorden(2k-1)/sqrt(fam)
            S[2m-1,2k] 	= -real(faodd)
            S[2m,2k] = imag(faodd)
            S[2m-1,2k-1] = real(faeven)
            S[2m,2k-1] = -imag(faeven)
        end
	end
	λ, S
end
