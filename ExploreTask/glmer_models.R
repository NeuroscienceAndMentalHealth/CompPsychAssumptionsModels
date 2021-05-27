library('lme4')

#this is assuming data is already scaled/z transformed

model2<-glmer(formula = choice ~ -1 + V + (-1 + V|participant_id),  
            data = gershman_data,
            family = binomial('probit'))


model3<-glmer(formula = choice ~ -1 + VTU + (-1 + VTU|participant_id),  
            data=gershman_data, 
            family = binomial('probit'))

model4<-glmer(formula = choice ~ -1 + V + RU + (-1 + V + RU|participant_id),  
            data=gershman_data, 
            family = binomial('probit'))


model5<-glmer(formula = choice ~ -1 + V + RU + VTU + (-1 + V + RU + VTU|participant_id),  
            data=gershman_data, 
            family = binomial('probit'))