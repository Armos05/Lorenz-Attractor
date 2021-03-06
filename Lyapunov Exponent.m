dom = [0,30];
N = chebop(@(t,x,y,z) [ diff(x) - 10*(y - x);
diff(y) - 28*x + y + x*z;
diff(z) + 8*z/3 - x*y ], dom);
ep = 1e-9;
N.lbc = @(x,y,z) [x+2; y+3; z-14];
[x1,y1,z1] = N\0; # Components of 1st trajectory
N.lbc = @(x,y,z) [x+2; y+3; z-14+ep];
[x2,y2,z2] = N\0; # Components of 2nd trajectory
d = sqrt(abs(x1-x2)^2 + abs(y1-y2)^2 + abs(z1-z2)^2);
semilogy(d)
xlabel('time')
title('magnitude of separation of nearby Lorenz trajectories')
hold on
x = chebfun('x', [0 dom(2)]);
semilogy(.8e-9 * exp(slope*x), 'k--')
legend('dist(traj_1, traj_2)', sprintf('exp(%1.2f x)', slope), ...
'location', 'northwest')
