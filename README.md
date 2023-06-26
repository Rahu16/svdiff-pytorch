# svdiff-pytorch
Repository for Assignment

Assignment was to build a FastAPI which will take a prompt as input request and return reponse after processing it through bark repo. As it was not clear what will be the output format of the image

I built two APIs

End Point of 1st API is /predict_image which will return image as a reponse output
End Point of 2nd API is /predict_image/byteString which will return byte string of the output image


In order to deploy it I have used Dockerfile and docker-compose. Application will run on 3000 port both internal and external port

As the model size is huge and it was taking long time for 25 inference steps in my local system, I decreased the inference step from 25 to 2 so that it can run quickly in my local


Probably on GPU based server it will run faster
So if you want to increase the inference steps to more than 2 then just set an environment variable "num_inference_steps" in docker-compose and redeploy it.


in order to deploy it 
1. Clone the git repo
2. Go to root folder of the repo
3. Execute sudo docker-compose up -d --build(docker and docker-compose has to be installed)

Your application will start running on port 3000
You can see the swagger documentation and test it on http://server-name:3000/docs


If you want to check the container logs, you can use
1. sudo docker logs <container-id> -f

When api is processing the text and returning the reponse it is saving the image