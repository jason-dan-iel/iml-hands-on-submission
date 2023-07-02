N = 50;
S = range(1., N, step=0.1);
o = range(0.1,0.9, length = 100);
L(S, o) = S*log(o) + (N-S)*log(1. -o);

# We have to install this package if not installed
using Plots
gr()

# MLE
p2 = Plots.heatmap(S,o, (S,o) -> L(S,o), color=:jet, xlabel="S", ylabel="θ", title="Bird's Eyes view");

# MLE at S = 25
SS=25;
vline!([SS],label=false,color=:black);
P3=Plots.plot(o,o->L(SS,o),label=false, xlabel='o',title="L(o|S=$SS)");

# Plotting
Plots.plot(p2,P3)
savefig("./pictures/mle-julia.png")