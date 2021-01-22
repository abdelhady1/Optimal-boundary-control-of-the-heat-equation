using JuMP, Plots, LinearAlgebra, SparseArrays, Ipopt

#Model initialization
mod = Model( optimizer_with_attributes(Ipopt.Optimizer , "max_iter" => 100,
                                        "mumps_mem_percent" => 500))
#The parameters of the 2-D body
λ = 45.0                              # Conductivity
c = 460.0                             # Specific heat capacitivity
ρ = 7800.0                            # Material density
α = λ / (c*ρ)                         # Diffusivity
L_y = 0.1                             #Width of rod, L_x = L_y / 3
N_y = 12                              #Discretization nodes in y-direction
N_x = Int(round(N_y / 3));            #Discretization nodes in x-direction
h = L_y / (N_y - 1)                   #step size Δx = Δy

#System , input and output matrices: θ' = Aθ(t) + Bu(t)
A = spdiagm(0 => -4*ones(N_x*N_y))
for i in 1:N_x * N_y
    if i % N_x == 1
        A[i+1,i] = 2
    elseif i % N_x == 0
        A[i-1,i] = 2
    else
        A[i-1,i] = 1
        A[i+1,i] = 1
    end
    if (i <= N_x)
        A[i+N_x,i] = 2
    elseif N_x * N_y - i < N_x
        A[i- N_x,i] = 2
    else
        A[i-N_x,i] = 1
        A[i+N_x,i] = 1
    end
end

A *= α / (h^2);

B = spzeros(3,N_x*N_y);
B[1,1] = 1;
B[2,2] = 1;
B[3,3] = 1;
B *= (2/ (c * ρ * h));

#y(t) = Cθ(t)
C = spzeros(N_x*N_y,1)
C[N_x*N_y] = 1.0;

#Integration from t0 = 0 to tf
tf = 30000.0;                   #Final time
dt = 1.5;                       #Sampling period Δt
steps = round(Int,tf / dt);     #number of time steps in simulation.

u_min = 0.0                     #lower input bound
u_max = 15000.0                 #upper input bound
@variable(mod,θ1[1:N_x*N_y,1:steps]);                                       # state vector plate1
@variable(mod,θ2[1:N_x*N_y,1:steps]);                                       # state vector plate2
@variable(mod,θ3[1:N_x*N_y,1:steps]);                                       # state vector plate3
@variable(mod,u_min <= u[i = 1:3,k = 1:steps - 1] <= u_max);                # Bounded Input u

θinit = 273.0                  #initial temperature of the body
θref = [400,600,500]           #Reference temperature for each plate

#Initial value added as constraints
@constraint(mod, θ1[:,1] .== θinit)
@constraint(mod, θ2[:,1] .== θinit)
@constraint(mod, θ3[:,1] .== θinit)
#System dynamics
for k in 1:steps - 1,i in 1:N_x*N_y
    @constraint(mod,θ1[i,k + 1] .==  θ1[i,k] + dt * ((A[:,i]' * θ1[:,k]) + B[1,i]' * u[1,k]))
    @constraint(mod,θ2[i,k + 1] ==  θ2[i,k] + dt * ((A[:,i]' * θ2[:,k]) + B[2,i]' * u[2,k]))
    @constraint(mod,θ3[i,k + 1] ==  θ3[i,k] + dt * ((A[:,i]' * θ3[:,k]) + B[3,i]' * u[3,k]))
end
#Q and R weighing matrices
Q = 1000 * C * C';
R = 0.001 * I;
#Cost function
@NLobjective(mod, Min, 0.5 * dt * sum(sum(Q[i,i] *(θref[1] - θ1[i,k])^2 for i in 1:N_x*N_y)
+sum(Q[i,i] *(θref[2] - θ2[i,k])^2 for i in 1:N_x*N_y) + sum(Q[i,i] *(θref[3] - θ3[i,k])^2 for i in 1:N_x*N_y)
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
    states_θstart1[i] = JuMP.value(θ1[1,i])
    states_θstart2[i] = JuMP.value(θ2[2,i])
    states_θstart3[i] = JuMP.value(θ3[3,i])
    states_θend3[i] = JuMP.value(θ3[N_x*N_y,i])
    states_θend2[i] = JuMP.value(θ2[N_x*N_y,i])
    states_θend1[i] = JuMP.value(θ1[N_x*N_y,i])
    input_u1[i] = JuMP.value(u[1,i])
    input_u2[i] = JuMP.value(u[2,i])
    input_u3[i] = JuMP.value(u[3,i])
end

states_θend3[end] = JuMP.value(θ3[end,end])
states_θend2[end] = JuMP.value(θ2[end,end])
states_θend1[end] = JuMP.value(θ1[end,end])
states_θstart1[end] = JuMP.value(θ1[1,end])
states_θstart2[end] = JuMP.value(θ2[2,end])
states_θstart3[end] = JuMP.value(θ3[3,end])
#Plotting temperatures at the input nodes
p2 = plot(states_θstart1, label = "θ_in1", xlabel = "time[s]", ylabel = "Temperature[K]")
p2 = plot!(states_θstart2,label = "θ_in2")
p2 = plot!(states_θstart3,label = "θ_in3")
 savefig("lq_optimal_2D_distributed.png")
#Plotting temperatures at the output
p1 = plot(states_θend3,label = "output-3", xlabel = "time[s]", ylabel = "Temperature[K]")
p1 = plot!(states_θend2,label = "output-2")
p1 = plot!(states_θend1,label = "output-1")
savefig("lq_optimal_2D-output-bounded_distributed.png")
#Plotting the input signals
p3 = plot(input_u1,label = "in_1", xlabel = "time[s]", ylabel = "Input")
p3 = plot!(input_u2,label = "in_2")
p3 = plot!(input_u3,label = "in_3")
savefig("lq_optimal_2D_inputbounded_distributed.png")
