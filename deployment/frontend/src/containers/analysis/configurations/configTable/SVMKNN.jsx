import { Close } from "@mui/icons-material";
import {
    IconButton,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
} from "@mui/material";

const SVMKNN = ({ configurations, handleDelete }) => {
    return (
        <Table>
            <TableHead>
                <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>Sampling Probability Decay Factor</TableCell>
                    <TableCell>Flipping Probability</TableCell>
                    <TableCell>Flipping Tags</TableCell>
                    <TableCell />
                </TableRow>
            </TableHead>
            <TableBody>
                {configurations.map((config, key) => (
                    <TableRow key={key}>
                        <TableCell>{config.name}</TableCell>
                        <TableCell>
                            {config.generator_config?.sample_prob_decay_factor}
                        </TableCell>
                        <TableCell>
                            {config.generator_config?.flip_prob}
                        </TableCell>
                        <TableCell>
                            {config.generator_config?.flipping_tags.join(", ")}
                        </TableCell>
                        <TableCell>
                            <IconButton
                                color="error"
                                onClick={() => handleDelete(key)}
                            >
                                <Close />
                            </IconButton>
                        </TableCell>
                    </TableRow>
                ))}
            </TableBody>
        </Table>
    );
};

export default SVMKNN;
