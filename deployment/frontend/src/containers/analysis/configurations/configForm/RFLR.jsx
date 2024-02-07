import commonStyles from "@/styles/commonStyles";
import { TextField } from "@mui/material";
import styles from "./styles";
import FormButtons from "@/components/formButtons/FormButtons";

export const RFDefaultValues = {
    threshold_classifier: 0.493399999999838,
    max_iter: 50,
    time_maximum: 120,
};

export const LRDefaultValues = {
    threshold_classifier: 0.491799999999785,
    max_iter: 50,
    time_maximum: 120,
};

const RFLR = ({ handleSubmit, addConfig, register, errors, reset, close }) => {
    const onSubmit = (config) => {
        config.threshold_classifier = parseFloat(config.threshold_classifier);
        addConfig(config);
    };
    return (
        <form onSubmit={handleSubmit(onSubmit)} style={styles.modalRoot}>
            <TextField
                sx={commonStyles.text}
                label="Name"
                type="text"
                {...register("name", { required: "Name is required" })}
                helperText={errors.name && errors.name.message}
                error={errors.name ? true : false}
            />
            <TextField
                sx={commonStyles.text}
                label="Classification Threshold"
                type="number"
                inputProps={{ step: 0.000000000000001 }}
                {...register("threshold_classifier", {
                    required: "Classification Threshold is required",
                    min: 0,
                })}
                helperText={
                    errors.threshold_classifier &&
                    errors.threshold_classifier.message
                }
                error={errors.threshold_classifier ? true : false}
            />
            <TextField
                sx={commonStyles.text}
                label="Maximum Iterations"
                type="number"
                {...register("max_iter", {
                    required: "Maximum Iterations is required",
                })}
                helperText={errors.max_iter && errors.max_iter.message}
                error={errors.max_iter ? true : false}
            />
            <TextField
                sx={commonStyles.text}
                label="Maximum Time"
                type="number"
                {...register("time_maximum", {
                    required: "Maximum Time is required",
                })}
                helperText={errors.time_maximum && errors.time_maximum.message}
                error={errors.time_maximum ? true : false}
            />
            <FormButtons reset={reset} close={close} />
        </form>
    );
};

export default RFLR;
