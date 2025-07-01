% prism refraction k means polar lights optimization (PKPLO)

function [Best_pos,Convergence_curve]=PKPLO(N,MaxFEs,lb,ub,dim,fobj)

%% Initialization
FEs = 0;
it = 1;
fitness=inf*ones(N,1);
fitness_new=inf*ones(N,1);

X=initialization(N,dim,ub,lb);
V=ones(N,dim);
X_new=zeros(N,dim);

for i=1:N
    fitness(i)=fobj(X(i,:));
    FEs=FEs+1;
end

[fitness, SortOrder]=sort(fitness);
X=X(SortOrder,:);
Bestpos=X(1,:);
Bestscore=fitness(1);

Convergence_curve=[];
Convergence_curve(it)=Bestscore;

NP = N;
nvar = dim;
SearchAgents_no = N;

[value, index_min] = min(fitness);
[value, index_max] = max(fitness);
G_best_kmean = X(index_min,:);
G_worst_kmean = X(index_max,:);
centroid_matrix = rand(NP,nvar); % creare random initial centroids matrix
[~,Centroid_matrix] = kmeans(centroid_matrix,3);% create random centroids position
particle_kmean = X;
particle_kmean_cost = fitness;
divide_num = dim/4;
w = 2;

%% Main loop
while FEs <= MaxFEs
    
    k_a = 1 -(1/MaxFEs)*FEs;
    Mean_centroid = (Centroid_matrix(1,:) + Centroid_matrix(2,:) + Centroid_matrix(3,:))/3;
    
    X_sum=sum(X,1);
    X_mean=X_sum/N;
    w1=tansig((FEs/MaxFEs)^4);
    w2=exp(-(2*FEs/MaxFEs)^3);
    best_M = Bestpos;
  
    for i=1:N
        probability = exp(-((FEs - MaxFEs/2)^2) / (2 * (MaxFEs/8)^2));
        if rand > probability
            a=rand()/2+1;
            V(i,:)=1*exp((1-a)/100*FEs);
            LS=V(i,:);
            
            GS=Levy(dim).*(X_mean-X(i,:)+(lb+rand(1,dim)*(ub-lb))/2);
            X_new(i,:)=X(i,:)+(w1*LS+w2*GS).*rand(1,dim);
        else
            trend = randi(3);
            switch trend
                case 1
                    [d1,d2] =drift_control(FEs,k_a);
                    KX_new(i,:) = 2*G_best_kmean.*rand + d1.*abs(-G_best_kmean.*(rand(1,nvar))+ X(i,:))+...
                        d2.*abs(-Mean_centroid.*(rand(1,nvar)) + X(i,:));
                case 2
                    KX_new(i,:) =  2*X(i,:).*rand +  (rand(1,nvar)).*(G_best_kmean - (rand(1,nvar)).*X(i,:)) + ...
                        (rand(1,nvar)).*(Centroid_matrix(1,:) - (rand(1,nvar)).*X(i,:))+...
                        (rand(1,nvar)).*(Centroid_matrix(2,:) - (rand(1,nvar)).*X(i,:))+...
                        (rand(1,nvar)).*(Centroid_matrix(3,:) - (rand(1,nvar)).*X(i,:));
                case 3
                    KX_new(i,:) = 2*Mean_centroid.*rand  + (rand(1,nvar)).*(G_best_kmean-(rand(1,nvar)).*Mean_centroid);
            end
            X_new(i,:) = KX_new(i,:);
        end
    end
    
    E =sqrt(FEs/MaxFEs);
    A=randperm(N);

    incidence = X_new;
    EPS = 1e-6;
    decay = 0.09;
    AA = rand(N, 1) * 75 + 15;
    mu = 1 + abs(sin(0.0174533 * (AA + fitness) / 2) ./ (EPS + sin(0.0174533 * AA / 2)));


    for i=1:N
        for j=1:dim
            if rand > 0.9+0.1*(FEs/MaxFEs)
                emergence = fitness(i) - incidence(i, j) + AA(i);
                r1 = rand * 2 - 1;
                theta = r1 * sin(AA(i)) * sqrt(mu(i)^2 - sin(emergence * 0.0174533)^2) - (sin(emergence * 0.0174533) * cos(AA(i) * 0.0174533));
                theta = max(min(theta, 1), -1);
                X_new(i, j) = asin(theta) * 57.2958;
                AA = AA .* exp(-decay * FEs / MaxFEs);
            end
        end
    end

    for i=1:N
        for j=1:dim
            if (rand<0.05) && (rand<E)
                X_new(i,j)=X(i,j)+sin(rand*pi)*(X(i,j)-X(A(i),j));
            end
        end
        
        Flag4ub=X_new(i,:)>ub;
        Flag4lb=X_new(i,:)<lb;
        X_new(i,:)=(X_new(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        fitness_new(i)=fobj(X_new(i,:));
        FEs=FEs+1;
        if fitness_new(i)<fitness(i)
            X(i,:)=X_new(i,:);
            fitness(i)=fitness_new(i);
        end
    end


    [fitness, SortOrder]=sort(fitness);
    X=X(SortOrder,:);

    [~,Centroid_matrix] = kmeans(X,3);
    G_best_kmean = X(1,:);

    if fitness(1)<Bestscore
        Bestpos=X(1,:);
        Bestscore=fitness(1);
    end
    it = it + 1;
    Convergence_curve(it)=Bestscore;
    Best_pos=Bestpos;

    %% Log progress
    if mod(FEs, N * 100) == 0
        fprintf('FEs: %6d/%d, Best Fitness: %.4e, Best Pos: %s\n', ...
                FEs, MaxFEs, Bestscore, mat2str(Best_pos, 4));
    end
end

end

function o=Levy(d)
beta=1.5;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u=randn(1,d)*sigma;v=randn(1,d);
step=u./abs(v).^(1/beta);
o=step;
end

function [d1 d2] =drift_control(cycle,a)
phi1 =  2*sin(0.1*cycle*a)*a;
phi2 = -2*sin(0.1*cycle*a)*a;
d1 = (rand.*phi1 + rand.*phi2);
d2 = (rand.*phi1 + rand.*phi2);
end

function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
end