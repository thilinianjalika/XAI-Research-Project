export const AWS_ACCESS_KEY = process.env["NEXT_PUBLIC_AWS_ACCESS_KEY"];
export const AWS_SECRET_KEY = process.env["NEXT_PUBLIC_AWS_SECRET_KEY"];
export const AWS_REGION = process.env["NEXT_PUBLIC_AWS_REGION"];
export const AWS_LAMBDA_NAME = process.env["NEXT_PUBLIC_AWS_XAI_LAMBDA"];

export const MODEL_NAME_KNN = "knn";
export const MODEL_NAME_SVM = "svm";
export const MODEL_NAME_LR = "lr";
export const MODEL_NAME_RF = "rf";

export const STATUS_CODE_MAP = {
    200: "OK",
    400: "Bad Request",
    500: "Server Error",
};
