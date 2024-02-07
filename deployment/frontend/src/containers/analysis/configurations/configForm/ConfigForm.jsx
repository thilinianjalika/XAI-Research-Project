import { useForm } from "react-hook-form";
import { useEffect } from "react";
import {
    MODEL_NAME_KNN,
    MODEL_NAME_LR,
    MODEL_NAME_RF,
    MODEL_NAME_SVM,
} from "@/constants";
import SVMKNN, { defaultValues as SVMKNNDefaultValues } from "./SVMKNN";
import RFLR, { LRDefaultValues, RFDefaultValues } from "./RFLR";
import { Box } from "@mui/material";

const ConfigForm = ({ model, addConfig, close, displayModal }) => {
    let defaultValues = SVMKNNDefaultValues;
    if (model === MODEL_NAME_RF) defaultValues = RFDefaultValues;
    else if (model === MODEL_NAME_LR) defaultValues = LRDefaultValues;

    const {
        register,
        handleSubmit,
        control,
        formState: { errors },
        reset,
    } = useForm({
        defaultValues,
    });
    useEffect(() => {
        if (displayModal) reset();
    }, [displayModal]);

    if (model === MODEL_NAME_KNN || model === MODEL_NAME_SVM) {
        return (
            <SVMKNN
                handleSubmit={handleSubmit}
                addConfig={addConfig}
                register={register}
                control={control}
                errors={errors}
                reset={reset}
                close={close}
            />
        );
    } else if (model === MODEL_NAME_RF || model === MODEL_NAME_LR) {
        return (
            <RFLR
                handleSubmit={handleSubmit}
                addConfig={addConfig}
                register={register}
                errors={errors}
                reset={reset}
                close={close}
            />
        );
    } else {
        return <Box>unknown</Box>;
    }
};

export default ConfigForm;
