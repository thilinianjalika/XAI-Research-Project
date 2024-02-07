import commonStyles from "@/styles/commonStyles";
import styles from "./styles";
import { Autocomplete, TextField, Typography } from "@mui/material";
import { Controller } from "react-hook-form";
import { tags } from "../flippingTags";
import FormButtons from "@/components/formButtons/FormButtons";

export const defaultValues = {
    sample_prob_decay_factor: 0.2,
    flip_prob: 0.5,
};

const SVMKNN = ({
    handleSubmit,
    addConfig,
    register,
    errors,
    control,
    close,
    reset,
}) => {
    const onAddConfig = (newConfig) => {
        if (newConfig.flipping_tags)
            newConfig.flipping_tags = JSON.parse(newConfig.flipping_tags);
        const { name, ...generator_config } = newConfig;
        const formattedConfig = { name, generator_config };
        addConfig(formattedConfig);
    };
    return (
        <form onSubmit={handleSubmit(onAddConfig)} style={styles.modalRoot}>
            <TextField
                sx={commonStyles.text}
                label="Name"
                type="text"
                {...register("name", { required: "Name is required" })}
                helperText={errors.name && errors.name.message}
                error={errors.name ? true : false}
            />
            <Typography variant="h5">Generator Configurations</Typography>
            <TextField
                sx={commonStyles.text}
                label="Sampling Probability Decay Factor"
                type="number"
                inputProps={{ step: 0.000000000000001 }}
                {...register("sample_prob_decay_factor", {
                    required: "Sampling Probability Decay Factor is required",
                    min: 0,
                })}
                helperText={
                    errors.sample_prob_decay_factor &&
                    errors.sample_prob_decay_factor.message
                }
                error={errors.sample_prob_decay_factor ? true : false}
            />
            <TextField
                sx={commonStyles.text}
                label="Flipping Probability"
                type="number"
                inputProps={{ step: 0.000000000000001 }}
                {...register("flip_prob", {
                    required: "Flipping Probability is required",
                    min: 0,
                    max: 1,
                })}
                helperText={errors.flip_prob && errors.flip_prob.message}
                error={errors.flip_prob ? true : false}
            />
            <Controller
                name="flipping_tags"
                control={control}
                rules={{ required: "Flipping Tags are required" }}
                render={({ field }) => (
                    <Autocomplete
                        {...field}
                        onChange={(e, val) => {
                            field.onChange({
                                target: { value: JSON.stringify(val) },
                            });
                        }}
                        value={field.value ? JSON.parse(field.value) : []}
                        multiple
                        options={tags}
                        renderInput={(params) => (
                            <TextField
                                {...params}
                                label="Flipping Tags"
                                error={!!errors.flipping_tags}
                                helperText={errors.flipping_tags?.message}
                            />
                        )}
                    />
                )}
            />
            <FormButtons reset={reset} close={close} />
        </form>
    );
};

export default SVMKNN;
