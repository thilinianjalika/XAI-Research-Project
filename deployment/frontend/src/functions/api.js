import { lambdaClient } from "@/clients/aws";
import { AWS_LAMBDA_NAME, STATUS_CODE_MAP } from "@/constants";
import { InvokeCommand } from "@aws-sdk/client-lambda";

export const getSentiment = async (model_name, prompt) => {
    const payload = {
        task: "evaluation",
        payload: { model_name, texts: [prompt] },
    };
    const command = new InvokeCommand({
        FunctionName: AWS_LAMBDA_NAME,
        Payload: JSON.stringify(payload),
    });
    const { Payload } = await lambdaClient.send(command);
    const result = Buffer.from(Payload).toString();
    const parsedResult = JSON.parse(result);
    if (parsedResult.status === 200) {
        const response = {
            score: parsedResult.body.scores[0],
            prediction: parsedResult.body.predictions[0],
        };
        return response;
    } else {
        console.error(parsedResult.body);
        const bubble = STATUS_CODE_MAP[parsedResult.status];
        throw bubble;
    }
};

export const startTestCase = async ({
    model_name,
    prompt,
    variations,
    configurations,
}) => {
    const payload = {
        task: "analysis",
        payload: { model_name, prompt, variations, configurations },
    };
    const command = new InvokeCommand({
        FunctionName: AWS_LAMBDA_NAME,
        Payload: JSON.stringify(payload),
    });
    const { Payload } = await lambdaClient.send(command);
    const result = Buffer.from(Payload).toString();
    const parsedResult = JSON.parse(result);
    if (parsedResult.status === 200) {
        return parsedResult.body;
    } else {
        console.error(parsedResult.body);
        const bubble = STATUS_CODE_MAP[parsedResult.status];
        throw bubble;
    }
};
