using JuMP, Ipopt, Plots, LinearAlgebra, SparseArrays

#A JuMP model is initialized
mod = Model(optimizer_with_attributes( Ipopt.Optimizer ,  "max_iter" => 100,
            "mumps_mem_percent" => 500))

#The parameters of the 1-D rod
L = 0.1                                 #Length of rod
λ = 45.0                                # Conductivity
c = 460.0                               # Specific heat capacitivity
ρ = 7800.0                              # Material density
α = λ / (c*ρ)                           # Diffusivity

#Number of spatial discretization points and length of each
N = 11                                 #Discretization points / nodes
Δx = L / (N-1)                         # Δx = x[i+1] - x[i]

#Integration from t0 = 0 to tf
tf = 10000.0;                          #Final time
Δt = 10^(-1);                          #Sampling period
k = round(Int,tf / Δt);                #k = number of steps

#System's matrix: θ' = Aθ + Bu
A = spdiagm(-1 => ones( N-1), 0 => -2*ones(N), 1 => ones(N-1));
A[2,1] = 2.0;
A[end - 1,end] = 2.0;
A = (α/( Δx^2)) * A;

#input matrix
B = spzeros(N);
B[1] = 1;
B = (2/ (c * ρ * Δx)) * B;

#output matrix
C = spzeros(1,N)
C[1,end] = 1.0;

#Initial and reference temperatures in kelvin
θinit = 273.0
θref = 500.0
#bounded input values[u_min,u_max]
u_min = 0.0
u_max = 15000.0

#Decision variables of the optimization problem
@variable(mod,θ[1:N,1:k]);                              # temperature at x[1:N] at time step k
@variable(mod,u_min <= u[1:k - 1] <= u_max);            # bounded constrained Input u
@variable(mod, y[1:k]);                                 # Output y
#Initial values
@constraint(mod, θ[:,1] .== θinit)
@constraint(mod, y[1] .== θinit)
#System dynamics
for j in 1:k - 1,i in 1:N
    @constraint(mod,θ[i,j + 1] - (θ[i,j] + Δt * ((A[:,i]' * θ[:,j]) + B[i] * u[j])).== 0.0)
end

for j in 1:k - 1
    @constraint(mod,y[j + 1] .== (C * θ[:,j + 1]))
end
#Weighing matrices
Q = C' * C * 1000.0;
R = 0.0001;

#err = sum(e' Q e)
err= @NLexpression(mod,(sum(Q[i,i] * (θref - θ[i,j]) ^ 2
                        for j in 1:k-1,i in 1:N)))
#in_err = sum(u'Ru)
in_err = @NLexpression(mod,(sum(R * u[j] ^ 2 for j in 1:k-1)))

J = @NLexpression(mod,0.5 * Δt * (err + in_err))
#defining the cost function
@NLobjective(mod, Min, J)
#optimizes the model
optimize!(mod)

#plotting values
states_θstart = zeros(k)
states_θend = zeros(k)
input_u = zeros(k - 1)
for i = 1 : k-1
    states_θstart[i] = JuMP.value(θ[1,i])
    states_θend[i] = JuMP.value(θ[end,i])
    input_u[i] = JuMP.value(u[i])
end

states_θend[end] = JuMP.value(θ[end,end])
states_θstart[end] = JuMP.value(θ[1,end])

plot(states_θstart,label = "θ1")
plot!(states_θend,label="y = Cθ")
savefig("lq-optimal_1D_Q1000_R0001_output-bounded.png")
p3 = plot(input_u,label="input,Q = 1000,R = 0.001")
savefig("lq-optimal_1D_Q1000_R0001-input-bounded.png")
