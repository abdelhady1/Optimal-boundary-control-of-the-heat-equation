using JuMP, Plots, LinearAlgebra, SparseArrays, Ipopt

#A JuMP model is initialized
mod = Model(optimizer_with_attributes(Ipopt.Optimizer ,  "max_iter" => 100,
                                        "mumps_mem_percent" => 500))
#The parameters of the 2-D body
λ = 45.0                            # Conductivity
c = 460.0                           # Specific heat capacitivity
ρ = 7800.0                          # Material density
L = 0.1                             #Length and width of body are assumed equivalent
α = λ / (c*ρ)                       # Diffusivity
N = 11                              # Number of Discretization points N_x = N_y
h = L / (N - 1)                     #Δx = Δy = h


#System , input and output matrices: θ' = Aθ(t) + Bu(t)
A = spdiagm(0 => -4*ones(N^2))
for i in 1:N^2
    if i%N == 1
        A[i+1,i] = 2
    elseif i%N == 0
        A[i-1,i] = 2
    else
        A[i-1,i] = 1
        A[i+1,i] = 1
    end
    if (i <= N)
        A[i+N,i] = 2
    elseif N^2 - i < N
        A[i- N,i] = 2
    else
        A[i-N,i] = 1
        A[i+N,i] = 1
    end
end
A *= α / (h^2);

B = spzeros(3,N^2);
B[1,1] = 1;
B[2,6] = 1;
B[3,11] = 1;
B *= (2/ (c * ρ * h));
#y(t) = Cθ(t)
C = spzeros(N^2,1)
C[111] = 1.0;
C[115] = 1.0;
C[end] = 1.0;
#Integration from t0 = 0 to tf
tf = 40000.0;                   #Final time
dt = 1.6;                       #Sampling period Δt
steps = round(Int,tf / dt);     #number of time steps in simulation.

u_min = 0.0                     #lower input bound
u_max = 15000.0                 #upper input bound
@variable(mod,θ[1:N^2,1:steps]);                                      # state vector
@variable(mod,u_min <= u[i = 1:3,k = 1:steps - 1] <= u_max);          # Bounded Input u

θinit = 273.0                           #initial temperature of the body
θref = 500.0                            #Reference temperature

@constraint(mod, θ[:,1] .== θinit)      #Initial value added as constraints
#System dynamics
for k in 1:steps - 1,i in 1:N^2
    @constraint(mod,θ[i,k + 1] .==  θ[i,k] + dt * ((A[:,i]' * θ[:,k]) + B[:,i]' * u[:,k]))
end
#Q and R weighing matrices
Q = 10000 * C * C';
R = 0.0001 * I;
#Cost function
@NLobjective(mod, Min, 0.5 * dt * sum(sum(Q[i,i] *(θref - θ[i,k])^2 for i in 1:N^2)
+ sum(R[i,i] * u[i,k]^2 for i in 1:3) for k in 1:steps - 1))
#Optimization process
optimize!(mod)

#Plotting results
states_θstart1 = zeros(steps)
states_θstart2 = zeros(steps)
states_θstart3 = zeros(steps)
states_θend3 = zeros(steps)
states_θend2 = zeros(steps)
states_θend1 = zeros(steps)
input_u1 = zeros(steps - 1)
input_u2 = zeros(steps - 1)
input_u3 = zeros(steps - 1)
for i = 1 : steps-1
    states_θstart1[i] = JuMP.value(θ[1,i])
    states_θstart2[i] = JuMP.value(θ[6,i])
    states_θstart3[i] = JuMP.value(θ[11,i])
    states_θend3[i] = JuMP.value(θ[115,i])
    states_θend2[i] = JuMP.value(θ[85,i])
    states_θend1[i] = JuMP.value(θ[45,i])
    input_u1[i] = JuMP.value(u[1,i])
    input_u2[i] = JuMP.value(u[2,i])
    input_u3[i] = JuMP.value(u[3,i])
end

states_θend3[end] = JuMP.value(θ[end,end])
states_θend2[end] = JuMP.value(θ[end-5,end])
states_θend1[end] = JuMP.value(θ[end-10,end])
states_θstart1[end] = JuMP.value(θ[1,end])
states_θstart2[end] = JuMP.value(θ[6,end])
states_θstart3[end] = JuMP.value(θ[11,end])
#Plotting temperatures at the input nodes
p2 = plot(states_θstart1, label = "θ_in1",legend = false)
p2 = plot!(states_θstart2,label = "θ_in2",legend = false)
p2 = plot!(states_θstart3,label = "θ_in3",legend = false)
savefig("lq_optimal_2D.png")
#Plotting temperatures at the output
p1 = plot(states_θend3,label = "output-3",legend = false)
p1 = plot!(states_θend2,label = "output-2",legend = false)
p1 = plot!(states_θend1,label = "output-1",legend = false)
savefig("lq_optimal_2D-output-bounded.png")
#Plotting the input signals
p3 = plot(input_u1,label = "in_1",legend = false)
p3 = plot!(input_u2,label = "in_2",legend = false)
p3 = plot!(input_u3,label = "in_3",legend = false)
savefig("lq_optimal_2D_inputbounded.png")
