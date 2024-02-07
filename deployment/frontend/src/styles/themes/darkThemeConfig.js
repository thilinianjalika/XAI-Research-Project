import { deepMerge } from "@/functions/util";
import base from "./base";

const overWrite = {
    palette: {
        mode: "dark",
    },
};

export default deepMerge(base, overWrite);
