import { AWS_ACCESS_KEY, AWS_REGION, AWS_SECRET_KEY } from "@/constants";
import { LambdaClient } from "@aws-sdk/client-lambda";

const lambdaClient = new LambdaClient({
    region: AWS_REGION,
    credentials: {
        accessKeyId: AWS_ACCESS_KEY,
        secretAccessKey: AWS_SECRET_KEY,
    },
});

export { lambdaClient };
