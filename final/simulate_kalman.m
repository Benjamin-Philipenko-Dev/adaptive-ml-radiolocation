function mse = simulate_kalman( q , mean_variance_matrix , X_train , h_x , Y_train )

    rng(42);

    % Simulation environment parameters
    grid_size = 100;           % Size of grid (100m x 100m)
    sigma_noise = 2;           % Noise variance level
    base_stations = [25, 25;   % Base station location 1
                     25, 75;   % Base station location 2
                     75, 25;   % Base station location 3
                     75, 75];  % Base station location 4

    % Path loss model parameters
    L0 = -30;   % RSS at reference distance of 0m (perfect signal connection), in dBm
    alpha = 2; % Path loss exponent
    f = 2.4e9; % Frequency in Hz (2.4 GHz for WLAN/WiFi)
    c = 3e8;   % Speed of light in m/s

    % Initialize Random Waypoint Motion Model
    timeStep = 1;                               % Poll position every second
    simulationTime = 120;                       % Total simulation time
    currentPosition = grid_size * rand(1, 2);   % Initial random starting position
    stopPoint = grid_size * rand(1, 2);         % Initial random stop point
    speed = 1 + 2 * rand;
    
    % State-transition matrix (location and veloctiy)
    F = [1 0 timeStep 0;   
         0 1 0 timeStep;
         0 0 1 0;
         0 0 0 1];
    
    % Observation matrix
    H = [1 0 0 0;  % Only position is measured, so H matrix maps state to measurement
         0 1 0 0];
    
    % Covariance of process noise
    Q = diag(q);
    
    % Covariance of measurement noise
    R = mean_variance_matrix; % This value is the mean variance estimation calculated previously in evaluation process for PWE
    
    % Initial State Estimate (position and velocity)
    x_k = [currentPosition, 0, 0]'; % General Variance Estimation 
    
    % Initial error covariance
    P_k = [mean_variance_matrix, zeros(2, 2); 
           zeros(2, 2), diag([10, 10])];    % General Variance Estimation 
    
    % Initialize matrices to store the real and predicted paths for plotting later
    real_path = zeros( simulationTime , 2 );
    pwe_estimated_path = zeros( simulationTime , 2 );
    kalman_predicted_path = zeros( simulationTime , 2 );
    
    % Main simulation loop
    for t = 1 : timeStep : simulationTime
    
        % Save current real position
        real_path(t, :) = currentPosition;
    
        % Simulate RSS measurements
        rss_measurement = zeros(1, 4);
        for ii = 1 : 4
            distance_to_bs = norm( currentPosition - base_stations( ii , : ) ); % Distance to base station
            rss_measurement( ii ) = L0 - 10 * alpha * log10(4 * pi * distance_to_bs * f / c) + ( sigma_noise * randn );
        end
    
        % Estimate position using PWE
        weights_loc = gaussian_kernel( rss_measurement , X_train , h_x );
        estimated_position = sum( weights_loc .* Y_train.' , 2 ).'; % PWE-estimated location
        
        % Save the PWE estimated position
        pwe_estimated_path( t , : ) = estimated_position;
    
    
        % Kalman Filter Implementation - General Variance Estimation 
        % Predict the next state
        x_k_plus_1 = F * x_k;                   % Predicted state estimate
        P_k_plus_1 = F * P_k * F' + Q;          % Predicted covariance estimate
    
        % Measurement update (PWE-estimated location)
        z_k = estimated_position';
    
        % Kalman gain calculation
        S_k = H * P_k_plus_1 * H' + R;  % Innovation covariance
        K_k = P_k_plus_1 * H' / S_k;    % Kalman gain
    
        % Update the state estimate
        x_k = x_k_plus_1 + K_k * (z_k - H * x_k_plus_1);
        
        % Update the error covariance
        P_k = (eye(4) - K_k * H) * P_k_plus_1;
        
        % Save the predicted position
        kalman_predicted_path(t, :) = x_k( 1 : 2 );  % Predicted position from Kalman filter
    
    
        
        % If the distance to the current stop point is less than the current movement increment, 
        % jump directly to the stop point. Once there, generate a new stop point and speed, then 
        % wait for the next time step. 
        if norm(stopPoint - currentPosition) <= speed * timeStep
            currentPosition = stopPoint;         % Jump to stop point
            stopPoint = grid_size * rand(1, 2);  % Generate new stop point
            speed = 1 + 2 * rand;                % Generate new speed
        else
            % The current stop point has not been reached so continue
            % travelling towards it at the current set speed.
            direction = (stopPoint - currentPosition) / norm(stopPoint - currentPosition);
            currentPosition = currentPosition + speed * timeStep * direction;
        end
    end

    % Return MSE as the objective value
    mse = mean( sum( abs( real_path( : , 1:2 ) - kalman_predicted_path( : , 1 : 2 ) ).^2 , 2 ) );

end