function weights = gaussian_kernel(X_i, X_train, h)

[ N_i , M_i ]  = size( X_i );
[ N_train , M_train ] = size( X_train );
weights = zeros( N_i , N_train );

% Run through all signal to estimate weights
for nn = 1 : N_i

    % Find sum of squared distance from observation to all training 
    % observations normalized by kernel width.
    delta = sum( abs( ( X_i( nn , : ) - X_train ) / h ).^2 , 2 );

    % Compute exponential of sum.
    weights( nn , : ) = exp( -0.5 * delta );

    % Normalize weights for sum of unity.
    weights( nn , : ) = weights( nn , : ) / sum( weights( nn , : ) );

end