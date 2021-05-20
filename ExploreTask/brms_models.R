model2<-brm(formula = choice ~ V + (1|participant_id),  
            data = gershman_data,
            family = cumulative(probit),
            warmup = 500,
            iter = 2000,
            chains = 2,
            inits = "random",
            cores = 2,
            seed = 123,
            prior(normal(0, num_outcomes), class = Intercept))



model3<-brm(formula = choice ~ VTU + (1|participant_id),  
            data=gershman_data, 
            family = cumulative(probit),
            warmup = 500, 
            iter = 2000, 
            chains = 2, 
            inits= "random", 
            cores=2,
            seed = 123,
            prior(normal(0, num_outcomes), class = Intercept))


model4<-brm(formula = choice ~ V + RU + (1|participant_id),  
            data=gershman_data, 
            family = cumulative(probit),
            warmup = 500, 
            iter = 2000, 
            chains = 2, 
            inits= "random", 
            cores=2,
            seed = 123,
            prior(normal(0, num_outcomes), class = Intercept))


model5<-brm(formula = choice ~ V + RU + VTU + (1|participant_id),  
            data=gershman_data, 
            family = cumulative(probit),
            warmup = 500, 
            iter = 2000, 
            chains = 2, 
            inits= "random", 
            cores=2,
            seed = 123,
            prior(normal(0, num_outcomes), class = Intercept))